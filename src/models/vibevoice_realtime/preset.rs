use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::rc::Rc;

use candle_core::{DType, Device, Tensor};
use zip::ZipArchive;

use crate::config::ModelAsset;
use crate::error::TtsError;
use crate::models::vibevoice::generation::LayerKvCache;

use super::runtime::RealtimeDecoderState;

#[derive(Clone)]
pub struct VoicePresetBranch {
    pub state: RealtimeDecoderState,
}

#[derive(Clone)]
pub struct VoicePreset {
    pub lm: VoicePresetBranch,
    pub tts_lm: VoicePresetBranch,
    pub _neg_lm: VoicePresetBranch,
    pub neg_tts_lm: VoicePresetBranch,
}

impl VoicePreset {
    pub fn load(asset: &ModelAsset, device: &Device, dtype: DType) -> Result<Self, TtsError> {
        let bytes = asset.read_bytes()?;
        let mut archive = ZipArchive::new(Cursor::new(bytes.to_vec())).map_err(|error| {
            TtsError::ModelError(format!(
                "Failed to open VibeVoice Realtime voice preset archive {}: {error}",
                asset.display_name(),
            ))
        })?;
        let root_prefix = archive
            .file_names()
            .find_map(|name| name.strip_suffix("data.pkl").map(str::to_string))
            .ok_or_else(|| {
                TtsError::ModelError(format!(
                    "VibeVoice Realtime voice preset {} is missing data.pkl",
                    asset.display_name(),
                ))
            })?;

        let mut pickle_bytes = Vec::new();
        archive
            .by_name(&format!("{root_prefix}data.pkl"))
            .map_err(|error| {
                TtsError::ModelError(format!(
                    "Failed to read data.pkl from {}: {error}",
                    asset.display_name(),
                ))
            })?
            .read_to_end(&mut pickle_bytes)?;

        let root = PickleReader::new(&pickle_bytes).read()?;
        let outputs = root.into_top_level_outputs()?;

        Ok(Self {
            lm: build_branch(
                outputs.required("lm")?,
                &mut archive,
                &root_prefix,
                device,
                dtype,
            )?,
            tts_lm: build_branch(
                outputs.required("tts_lm")?,
                &mut archive,
                &root_prefix,
                device,
                dtype,
            )?,
            _neg_lm: build_branch(
                outputs.required("neg_lm")?,
                &mut archive,
                &root_prefix,
                device,
                dtype,
            )?,
            neg_tts_lm: build_branch(
                outputs.required("neg_tts_lm")?,
                &mut archive,
                &root_prefix,
                device,
                dtype,
            )?,
        })
    }
}

fn build_branch<R: Read + std::io::Seek>(
    output: BaseModelOutputDescriptor,
    archive: &mut ZipArchive<R>,
    root_prefix: &str,
    device: &Device,
    runtime_dtype: DType,
) -> Result<VoicePresetBranch, TtsError> {
    let last_hidden_state =
        output
            .last_hidden_state
            .materialize(archive, root_prefix, device, runtime_dtype)?;
    let prompt_len = last_hidden_state.dim(1)?;
    let last_hidden = last_hidden_state.narrow(1, prompt_len - 1, 1)?.squeeze(1)?;
    let layer_caches =
        output
            .past_key_values
            .into_layer_caches(archive, root_prefix, device, runtime_dtype)?;

    Ok(VoicePresetBranch {
        state: RealtimeDecoderState::new(prompt_len, last_hidden, layer_caches),
    })
}

#[derive(Clone)]
struct TopLevelOutputs {
    values: HashMap<String, BaseModelOutputDescriptor>,
}

impl TopLevelOutputs {
    fn required(&self, key: &str) -> Result<BaseModelOutputDescriptor, TtsError> {
        self.values.get(key).cloned().ok_or_else(|| {
            TtsError::ModelError(format!(
                "VibeVoice Realtime voice preset is missing the {key} branch"
            ))
        })
    }
}

#[derive(Clone)]
enum PickleValue {
    Mark,
    None,
    Bool,
    Int(i64),
    String(String),
    Tuple(Vec<PickleValue>),
    List(Rc<RefCell<Vec<PickleValue>>>),
    Dict(Vec<(PickleValue, PickleValue)>),
    OrderedDict(Vec<(PickleValue, PickleValue)>),
    Global(GlobalRef),
    Storage(StorageDescriptor),
    Tensor(TensorDescriptor),
    DynamicCache(Rc<RefCell<DynamicCacheDescriptor>>),
    BaseModelOutput(Rc<RefCell<BaseModelOutputDescriptor>>),
}

impl PickleValue {
    fn into_top_level_outputs(self) -> Result<TopLevelOutputs, TtsError> {
        let PickleValue::Dict(entries) = self else {
            return Err(TtsError::ModelError(
                "Expected a top-level dictionary in the VibeVoice Realtime preset".to_string(),
            ));
        };

        let mut values = HashMap::new();
        for (key, value) in entries {
            let key = key.into_string()?;
            let value = value.into_base_model_output()?;
            values.insert(key, value);
        }
        Ok(TopLevelOutputs { values })
    }

    fn into_string(self) -> Result<String, TtsError> {
        match self {
            Self::String(value) => Ok(value),
            _ => Err(TtsError::ModelError(
                "Expected a string while decoding a VibeVoice Realtime preset".to_string(),
            )),
        }
    }

    fn into_int(self) -> Result<i64, TtsError> {
        match self {
            Self::Int(value) => Ok(value),
            _ => Err(TtsError::ModelError(
                "Expected an integer while decoding a VibeVoice Realtime preset".to_string(),
            )),
        }
    }

    fn into_tuple(self) -> Result<Vec<PickleValue>, TtsError> {
        match self {
            Self::Tuple(values) => Ok(values),
            _ => Err(TtsError::ModelError(
                "Expected a tuple while decoding a VibeVoice Realtime preset".to_string(),
            )),
        }
    }

    fn into_list(self) -> Result<Vec<PickleValue>, TtsError> {
        match self {
            Self::List(values) => Ok(values.borrow().clone()),
            _ => Err(TtsError::ModelError(
                "Expected a list while decoding a VibeVoice Realtime preset".to_string(),
            )),
        }
    }

    fn into_dict(self) -> Result<Vec<(PickleValue, PickleValue)>, TtsError> {
        match self {
            Self::Dict(values) | Self::OrderedDict(values) => Ok(values),
            _ => Err(TtsError::ModelError(
                "Expected a dictionary while decoding a VibeVoice Realtime preset".to_string(),
            )),
        }
    }

    fn into_global(self) -> Result<GlobalRef, TtsError> {
        match self {
            Self::Global(value) => Ok(value),
            _ => Err(TtsError::ModelError(
                "Expected a callable while decoding a VibeVoice Realtime preset".to_string(),
            )),
        }
    }

    fn into_base_model_output(self) -> Result<BaseModelOutputDescriptor, TtsError> {
        match self {
            Self::BaseModelOutput(value) => Ok(value.borrow().clone()),
            _ => Err(TtsError::ModelError(
                "Expected a BaseModelOutputWithPast object in the VibeVoice Realtime preset"
                    .to_string(),
            )),
        }
    }
}

#[derive(Clone)]
enum GlobalRef {
    BaseModelOutputWithPast,
    RebuildTensorV2,
    BFloat16Storage,
    OrderedDict,
    DynamicCache,
}

#[derive(Clone)]
struct StorageDescriptor {
    storage_id: String,
    dtype: DType,
    size: usize,
}

#[derive(Clone)]
struct TensorDescriptor {
    storage: StorageDescriptor,
    storage_offset: usize,
    size: Vec<usize>,
    stride: Vec<usize>,
}

impl TensorDescriptor {
    fn materialize<R: Read + std::io::Seek>(
        &self,
        archive: &mut ZipArchive<R>,
        root_prefix: &str,
        device: &Device,
        runtime_dtype: DType,
    ) -> Result<Tensor, TtsError> {
        let storage_bytes = read_storage_bytes(archive, root_prefix, &self.storage)?;
        let bytes = materialize_strided_bytes(
            &storage_bytes,
            self.storage.dtype,
            self.storage_offset,
            &self.size,
            &self.stride,
        )?;
        let tensor =
            Tensor::from_raw_buffer(bytes.as_slice(), self.storage.dtype, &self.size, device)?;
        if tensor.dtype() != runtime_dtype {
            tensor.to_dtype(runtime_dtype).map_err(Into::into)
        } else {
            Ok(tensor)
        }
    }
}

#[derive(Clone)]
struct DynamicCacheDescriptor {
    seen_tokens: usize,
    key_cache: Vec<TensorDescriptor>,
    value_cache: Vec<TensorDescriptor>,
}

impl DynamicCacheDescriptor {
    fn into_layer_caches<R: Read + std::io::Seek>(
        self,
        archive: &mut ZipArchive<R>,
        root_prefix: &str,
        device: &Device,
        runtime_dtype: DType,
    ) -> Result<Vec<LayerKvCache>, TtsError> {
        if self.key_cache.len() != self.value_cache.len() {
            return Err(TtsError::ModelError(
                "VibeVoice Realtime preset cache had mismatched key/value lengths".to_string(),
            ));
        }

        let mut caches = Vec::with_capacity(self.key_cache.len());
        for (key, value) in self.key_cache.into_iter().zip(self.value_cache) {
            let key = key.materialize(archive, root_prefix, device, runtime_dtype)?;
            let value = value.materialize(archive, root_prefix, device, runtime_dtype)?;
            caches.push(Some(crate::layers::kv_cache::KvCacheState::from_tensors(
                key, value,
            )?));
        }
        Ok(caches)
    }
}

#[derive(Clone)]
struct BaseModelOutputDescriptor {
    last_hidden_state: TensorDescriptor,
    past_key_values: DynamicCacheDescriptor,
}

struct PickleReader<'a> {
    bytes: &'a [u8],
    position: usize,
    stack: Vec<PickleValue>,
    memo: HashMap<u32, PickleValue>,
}

impl<'a> PickleReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            position: 0,
            stack: Vec::new(),
            memo: HashMap::new(),
        }
    }

    fn read(mut self) -> Result<PickleValue, TtsError> {
        while self.position < self.bytes.len() {
            let opcode = self.read_u8()?;
            match opcode {
                0x80 => {
                    self.read_u8()?;
                }
                b'.' => {
                    return self.pop();
                }
                b'(' => self.stack.push(PickleValue::Mark),
                b'}' => self.stack.push(PickleValue::Dict(Vec::new())),
                b']' => self
                    .stack
                    .push(PickleValue::List(Rc::new(RefCell::new(Vec::new())))),
                b')' => self.stack.push(PickleValue::Tuple(Vec::new())),
                b'N' => self.stack.push(PickleValue::None),
                0x88 => self.stack.push(PickleValue::Bool),
                0x89 => self.stack.push(PickleValue::Bool),
                b'K' => {
                    let value = self.read_u8()? as i64;
                    self.stack.push(PickleValue::Int(value));
                }
                b'M' => {
                    let value = self.read_u16_le()? as i64;
                    self.stack.push(PickleValue::Int(value));
                }
                b'J' => {
                    let value = self.read_i32_le()? as i64;
                    self.stack.push(PickleValue::Int(value));
                }
                b'X' => {
                    let value = self.read_binunicode()?;
                    self.stack.push(PickleValue::String(value));
                }
                0x8c => {
                    let value = self.read_short_binunicode()?;
                    self.stack.push(PickleValue::String(value));
                }
                b'c' => {
                    let value = self.read_global()?;
                    self.stack.push(PickleValue::Global(value));
                }
                b'q' => {
                    let index = self.read_u8()? as u32;
                    let value = self.stack.last().cloned().ok_or_else(|| {
                        TtsError::ModelError(
                            "Attempted to memoize an empty pickle stack".to_string(),
                        )
                    })?;
                    self.memo.insert(index, value);
                }
                b'r' => {
                    let index = self.read_u32_le()?;
                    let value = self.stack.last().cloned().ok_or_else(|| {
                        TtsError::ModelError(
                            "Attempted to memoize an empty pickle stack".to_string(),
                        )
                    })?;
                    self.memo.insert(index, value);
                }
                b'h' => {
                    let index = self.read_u8()? as u32;
                    let value = self.memo.get(&index).cloned().ok_or_else(|| {
                        TtsError::ModelError(format!(
                            "Unknown BINGET index {index} in VibeVoice Realtime preset"
                        ))
                    })?;
                    self.stack.push(value);
                }
                b'j' => {
                    let index = self.read_u32_le()?;
                    let value = self.memo.get(&index).cloned().ok_or_else(|| {
                        TtsError::ModelError(format!(
                            "Unknown LONG_BINGET index {index} in VibeVoice Realtime preset"
                        ))
                    })?;
                    self.stack.push(value);
                }
                b't' => {
                    let values = self.pop_mark_items()?;
                    self.stack.push(PickleValue::Tuple(values));
                }
                0x85 => {
                    let value = self.pop()?;
                    self.stack.push(PickleValue::Tuple(vec![value]));
                }
                0x86 => {
                    let right = self.pop()?;
                    let left = self.pop()?;
                    self.stack.push(PickleValue::Tuple(vec![left, right]));
                }
                0x87 => {
                    let third = self.pop()?;
                    let second = self.pop()?;
                    let first = self.pop()?;
                    self.stack
                        .push(PickleValue::Tuple(vec![first, second, third]));
                }
                b'e' => {
                    let values = self.pop_mark_items()?;
                    let list = self.pop()?;
                    self.stack.push(apply_appends(list, values)?);
                }
                b'u' => {
                    let items = self.pop_mark_items()?;
                    let object = self.pop()?;
                    self.stack.push(apply_setitems(object, items)?);
                }
                b'R' => {
                    let args = self.pop()?.into_tuple()?;
                    let callable = self.pop()?.into_global()?;
                    self.stack.push(apply_reduce(callable, args)?);
                }
                0x81 => {
                    let args = self.pop()?.into_tuple()?;
                    let class = self.pop()?.into_global()?;
                    self.stack.push(apply_newobj(class, args)?);
                }
                b'b' => {
                    let state = self.pop()?;
                    let object = self.pop()?;
                    self.stack.push(apply_build(object, state)?);
                }
                b'Q' => {
                    let pid = self.pop()?;
                    self.stack
                        .push(PickleValue::Storage(parse_persistent_id(pid)?));
                }
                other => {
                    return Err(TtsError::ModelError(format!(
                        "Unsupported pickle opcode 0x{other:02x} in VibeVoice Realtime preset"
                    )));
                }
            }
        }

        Err(TtsError::ModelError(
            "Unexpected end of VibeVoice Realtime preset pickle".to_string(),
        ))
    }

    fn pop(&mut self) -> Result<PickleValue, TtsError> {
        self.stack.pop().ok_or_else(|| {
            TtsError::ModelError(
                "Unexpected empty pickle stack in VibeVoice Realtime preset".to_string(),
            )
        })
    }

    fn pop_mark_items(&mut self) -> Result<Vec<PickleValue>, TtsError> {
        let Some(mark_index) = self
            .stack
            .iter()
            .rposition(|value| matches!(value, PickleValue::Mark))
        else {
            return Err(TtsError::ModelError(
                "Missing pickle MARK in VibeVoice Realtime preset".to_string(),
            ));
        };
        let items = self.stack.split_off(mark_index + 1);
        self.stack.pop();
        Ok(items)
    }

    fn read_u8(&mut self) -> Result<u8, TtsError> {
        let value = *self.bytes.get(self.position).ok_or_else(|| {
            TtsError::ModelError("Unexpected end of VibeVoice Realtime preset bytes".to_string())
        })?;
        self.position += 1;
        Ok(value)
    }

    fn read_u16_le(&mut self) -> Result<u16, TtsError> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i32_le(&mut self) -> Result<i32, TtsError> {
        let bytes = self.read_exact(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u32_le(&mut self) -> Result<u32, TtsError> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8], TtsError> {
        let end = self.position + len;
        let bytes = self.bytes.get(self.position..end).ok_or_else(|| {
            TtsError::ModelError("Unexpected end of VibeVoice Realtime preset bytes".to_string())
        })?;
        self.position = end;
        Ok(bytes)
    }

    fn read_binunicode(&mut self) -> Result<String, TtsError> {
        let len = self.read_u32_le()? as usize;
        self.read_utf8(len)
    }

    fn read_short_binunicode(&mut self) -> Result<String, TtsError> {
        let len = self.read_u8()? as usize;
        self.read_utf8(len)
    }

    fn read_utf8(&mut self, len: usize) -> Result<String, TtsError> {
        std::str::from_utf8(self.read_exact(len)?)
            .map(|value| value.to_string())
            .map_err(|error| {
                TtsError::ModelError(format!(
                    "Invalid UTF-8 in VibeVoice Realtime preset pickle: {error}"
                ))
            })
    }

    fn read_global(&mut self) -> Result<GlobalRef, TtsError> {
        let module = self.read_line()?;
        let name = self.read_line()?;
        match (module.as_str(), name.as_str()) {
            ("transformers.modeling_outputs", "BaseModelOutputWithPast") => {
                Ok(GlobalRef::BaseModelOutputWithPast)
            }
            ("torch._utils", "_rebuild_tensor_v2") => Ok(GlobalRef::RebuildTensorV2),
            ("torch", "BFloat16Storage") => Ok(GlobalRef::BFloat16Storage),
            ("collections", "OrderedDict") => Ok(GlobalRef::OrderedDict),
            ("transformers.cache_utils", "DynamicCache") => Ok(GlobalRef::DynamicCache),
            _ => Err(TtsError::ModelError(format!(
                "Unsupported pickle GLOBAL {module}.{name} in VibeVoice Realtime preset"
            ))),
        }
    }

    fn read_line(&mut self) -> Result<String, TtsError> {
        let start = self.position;
        while self.position < self.bytes.len() && self.bytes[self.position] != b'\n' {
            self.position += 1;
        }
        if self.position >= self.bytes.len() {
            return Err(TtsError::ModelError(
                "Unexpected end of VibeVoice Realtime preset while reading GLOBAL".to_string(),
            ));
        }
        let line = std::str::from_utf8(&self.bytes[start..self.position])
            .map_err(|error| {
                TtsError::ModelError(format!(
                    "Invalid UTF-8 in VibeVoice Realtime preset GLOBAL: {error}"
                ))
            })?
            .to_string();
        self.position += 1;
        Ok(line)
    }
}

fn apply_reduce(callable: GlobalRef, args: Vec<PickleValue>) -> Result<PickleValue, TtsError> {
    match callable {
        GlobalRef::OrderedDict => Ok(PickleValue::OrderedDict(Vec::new())),
        GlobalRef::RebuildTensorV2 => {
            if args.len() < 5 {
                return Err(TtsError::ModelError(
                    "Malformed tensor descriptor in VibeVoice Realtime preset".to_string(),
                ));
            }
            let storage = match args[0].clone() {
                PickleValue::Storage(storage) => storage,
                _ => {
                    return Err(TtsError::ModelError(
                        "Expected tensor storage descriptor in VibeVoice Realtime preset"
                            .to_string(),
                    ))
                }
            };
            let storage_offset = args[1].clone().into_int()? as usize;
            let size = args[2]
                .clone()
                .into_tuple()?
                .into_iter()
                .map(|value| Ok(value.into_int()? as usize))
                .collect::<Result<Vec<_>, TtsError>>()?;
            let stride = args[3]
                .clone()
                .into_tuple()?
                .into_iter()
                .map(|value| Ok(value.into_int()? as usize))
                .collect::<Result<Vec<_>, TtsError>>()?;

            Ok(PickleValue::Tensor(TensorDescriptor {
                storage,
                storage_offset,
                size,
                stride,
            }))
        }
        GlobalRef::BaseModelOutputWithPast => {
            if args.len() < 2 {
                return Err(TtsError::ModelError(
                    "Malformed BaseModelOutputWithPast in VibeVoice Realtime preset".to_string(),
                ));
            }
            let last_hidden_state = match args[0].clone() {
                PickleValue::Tensor(tensor) => tensor,
                _ => {
                    return Err(TtsError::ModelError(
                        "Expected last_hidden_state tensor in VibeVoice Realtime preset"
                            .to_string(),
                    ))
                }
            };
            let past_key_values = match args[1].clone() {
                PickleValue::DynamicCache(cache) => cache.borrow().clone(),
                _ => {
                    return Err(TtsError::ModelError(
                        "Expected DynamicCache in VibeVoice Realtime preset".to_string(),
                    ))
                }
            };
            Ok(PickleValue::BaseModelOutput(Rc::new(RefCell::new(
                BaseModelOutputDescriptor {
                    last_hidden_state,
                    past_key_values,
                },
            ))))
        }
        other => Err(TtsError::ModelError(format!(
            "Unsupported REDUCE callable in VibeVoice Realtime preset: {}",
            describe_global(&other)
        ))),
    }
}

fn apply_newobj(class: GlobalRef, args: Vec<PickleValue>) -> Result<PickleValue, TtsError> {
    match class {
        GlobalRef::DynamicCache => {
            if !args.is_empty() {
                return Err(TtsError::ModelError(
                    "Unexpected DynamicCache constructor arguments in VibeVoice Realtime preset"
                        .to_string(),
                ));
            }
            Ok(PickleValue::DynamicCache(Rc::new(RefCell::new(
                DynamicCacheDescriptor {
                    seen_tokens: 0,
                    key_cache: Vec::new(),
                    value_cache: Vec::new(),
                },
            ))))
        }
        other => Err(TtsError::ModelError(format!(
            "Unsupported NEWOBJ class in VibeVoice Realtime preset: {}",
            describe_global(&other)
        ))),
    }
}

fn apply_build(object: PickleValue, state: PickleValue) -> Result<PickleValue, TtsError> {
    match object {
        PickleValue::DynamicCache(cache) => {
            let mut cache_ref = cache.borrow_mut();
            for (key, value) in state.into_dict()? {
                match key.into_string()?.as_str() {
                    "_seen_tokens" => cache_ref.seen_tokens = value.into_int()? as usize,
                    "key_cache" => {
                        cache_ref.key_cache = value
                            .into_list()?
                            .into_iter()
                            .map(|value| match value {
                                PickleValue::Tensor(tensor) => Ok(tensor),
                                _ => Err(TtsError::ModelError(
                                    "Expected a tensor in DynamicCache.key_cache".to_string(),
                                )),
                            })
                            .collect::<Result<Vec<_>, _>>()?
                    }
                    "value_cache" => {
                        cache_ref.value_cache = value
                            .into_list()?
                            .into_iter()
                            .map(|value| match value {
                                PickleValue::Tensor(tensor) => Ok(tensor),
                                _ => Err(TtsError::ModelError(
                                    "Expected a tensor in DynamicCache.value_cache".to_string(),
                                )),
                            })
                            .collect::<Result<Vec<_>, _>>()?
                    }
                    _ => {}
                }
            }
            drop(cache_ref);
            Ok(PickleValue::DynamicCache(cache))
        }
        PickleValue::BaseModelOutput(output) => {
            let mut output_ref = output.borrow_mut();
            for (key, value) in state.into_dict()? {
                match key.into_string()?.as_str() {
                    "last_hidden_state" => {
                        output_ref.last_hidden_state = match value {
                            PickleValue::Tensor(tensor) => tensor,
                            _ => {
                                return Err(TtsError::ModelError(
                                    "Expected last_hidden_state tensor while BUILDing BaseModelOutputWithPast"
                                        .to_string(),
                                ))
                            }
                        }
                    }
                    "past_key_values" => {
                        output_ref.past_key_values = match value {
                            PickleValue::DynamicCache(cache) => cache.borrow().clone(),
                            _ => {
                                return Err(TtsError::ModelError(
                                    "Expected DynamicCache while BUILDing BaseModelOutputWithPast"
                                        .to_string(),
                                ))
                            }
                        }
                    }
                    _ => {}
                }
            }
            drop(output_ref);
            Ok(PickleValue::BaseModelOutput(output))
        }
        other => Ok(other),
    }
}

fn apply_setitems(object: PickleValue, items: Vec<PickleValue>) -> Result<PickleValue, TtsError> {
    let entries = items
        .chunks_exact(2)
        .map(|chunk| (chunk[0].clone(), chunk[1].clone()))
        .collect::<Vec<_>>();

    match object {
        PickleValue::Dict(mut values) => {
            values.extend(entries);
            Ok(PickleValue::Dict(values))
        }
        PickleValue::OrderedDict(mut values) => {
            values.extend(entries);
            Ok(PickleValue::OrderedDict(values))
        }
        PickleValue::BaseModelOutput(output) => {
            let mut output_ref = output.borrow_mut();
            for (key, value) in entries {
                match key.into_string()?.as_str() {
                    "last_hidden_state" => {
                        output_ref.last_hidden_state = match value {
                            PickleValue::Tensor(tensor) => tensor,
                            _ => {
                                return Err(TtsError::ModelError(
                                    "Expected last_hidden_state tensor while applying SETITEMS to BaseModelOutputWithPast"
                                        .to_string(),
                                ))
                            }
                        }
                    }
                    "past_key_values" => {
                        output_ref.past_key_values = match value {
                            PickleValue::DynamicCache(cache) => cache.borrow().clone(),
                            _ => {
                                return Err(TtsError::ModelError(
                                    "Expected DynamicCache while applying SETITEMS to BaseModelOutputWithPast"
                                        .to_string(),
                                ))
                            }
                        }
                    }
                    _ => {}
                }
            }
            drop(output_ref);
            Ok(PickleValue::BaseModelOutput(output))
        }
        other => Err(TtsError::ModelError(format!(
            "SETITEMS target was not a supported mapping in VibeVoice Realtime preset: {}",
            describe_pickle_value(&other)
        ))),
    }
}

fn apply_appends(object: PickleValue, values: Vec<PickleValue>) -> Result<PickleValue, TtsError> {
    match object {
        PickleValue::List(list) => {
            list.borrow_mut().extend(values);
            Ok(PickleValue::List(list))
        }
        other => Err(TtsError::ModelError(format!(
            "APPENDS target was not a list in VibeVoice Realtime preset: {}",
            describe_pickle_value(&other)
        ))),
    }
}

fn parse_persistent_id(value: PickleValue) -> Result<StorageDescriptor, TtsError> {
    let values = value.into_tuple()?;
    if values.len() < 5 {
        return Err(TtsError::ModelError(
            "Malformed persistent storage reference in VibeVoice Realtime preset".to_string(),
        ));
    }
    let storage_tag = values[0].clone().into_string()?;
    if storage_tag != "storage" {
        return Err(TtsError::ModelError(format!(
            "Unsupported persistent storage tag {storage_tag} in VibeVoice Realtime preset"
        )));
    }
    let dtype = match values[1].clone().into_global()? {
        GlobalRef::BFloat16Storage => DType::BF16,
        other => {
            return Err(TtsError::ModelError(format!(
                "Unsupported storage dtype in VibeVoice Realtime preset: {}",
                describe_global(&other)
            )))
        }
    };
    let storage_id = values[2].clone().into_string()?;
    let size = values[4].clone().into_int()? as usize;
    Ok(StorageDescriptor {
        storage_id,
        dtype,
        size,
    })
}

fn describe_global(value: &GlobalRef) -> &'static str {
    match value {
        GlobalRef::BaseModelOutputWithPast => "BaseModelOutputWithPast",
        GlobalRef::RebuildTensorV2 => "_rebuild_tensor_v2",
        GlobalRef::BFloat16Storage => "BFloat16Storage",
        GlobalRef::OrderedDict => "OrderedDict",
        GlobalRef::DynamicCache => "DynamicCache",
    }
}

fn describe_pickle_value(value: &PickleValue) -> &'static str {
    match value {
        PickleValue::Mark => "mark",
        PickleValue::None => "none",
        PickleValue::Bool => "bool",
        PickleValue::Int(_) => "int",
        PickleValue::String(_) => "string",
        PickleValue::Tuple(_) => "tuple",
        PickleValue::List(_) => "list",
        PickleValue::Dict(_) => "dict",
        PickleValue::OrderedDict(_) => "ordered-dict",
        PickleValue::Global(_) => "global",
        PickleValue::Storage(_) => "storage",
        PickleValue::Tensor(_) => "tensor",
        PickleValue::DynamicCache(_) => "dynamic-cache",
        PickleValue::BaseModelOutput(_) => "base-model-output",
    }
}

fn read_storage_bytes<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    root_prefix: &str,
    storage: &StorageDescriptor,
) -> Result<Vec<u8>, TtsError> {
    let mut bytes = Vec::new();
    archive
        .by_name(&format!("{root_prefix}data/{}", storage.storage_id))
        .map_err(|error| {
            TtsError::ModelError(format!(
                "Failed to read storage {} from VibeVoice Realtime preset: {error}",
                storage.storage_id,
            ))
        })?
        .read_to_end(&mut bytes)?;
    let expected = storage.size * dtype_size(storage.dtype);
    if bytes.len() < expected {
        return Err(TtsError::ModelError(format!(
            "Storage {} in VibeVoice Realtime preset had {} bytes, expected at least {}",
            storage.storage_id,
            bytes.len(),
            expected,
        )));
    }
    Ok(bytes)
}

fn materialize_strided_bytes(
    storage_bytes: &[u8],
    dtype: DType,
    storage_offset: usize,
    size: &[usize],
    stride: &[usize],
) -> Result<Vec<u8>, TtsError> {
    if size.len() != stride.len() {
        return Err(TtsError::ModelError(
            "Tensor size/stride rank mismatch in VibeVoice Realtime preset".to_string(),
        ));
    }
    let elem_size = dtype_size(dtype);
    let elem_count = size.iter().product::<usize>();
    let mut output = vec![0u8; elem_count * elem_size];

    for linear_index in 0..elem_count {
        let coords = unravel_index(linear_index, size);
        let storage_index = storage_offset
            + coords
                .iter()
                .zip(stride.iter())
                .map(|(coord, stride)| coord * stride)
                .sum::<usize>();
        let source_start = storage_index * elem_size;
        let source_end = source_start + elem_size;
        let dest_start = linear_index * elem_size;
        let dest_end = dest_start + elem_size;
        output[dest_start..dest_end].copy_from_slice(
            storage_bytes.get(source_start..source_end).ok_or_else(|| {
                TtsError::ModelError(
                    "Tensor view exceeded storage bounds in VibeVoice Realtime preset".to_string(),
                )
            })?,
        );
    }

    Ok(output)
}

fn unravel_index(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0usize; shape.len()];
    for dim in (0..shape.len()).rev() {
        let extent = shape[dim].max(1);
        coords[dim] = index % extent;
        index /= extent;
    }
    coords
}

fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::BF16 | DType::F16 => 2,
        DType::F32 | DType::I32 | DType::U32 => 4,
        DType::F64 | DType::I64 => 8,
        DType::U8 => 1,
        _ => 2,
    }
}
