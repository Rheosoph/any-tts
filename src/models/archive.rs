use std::io::{Read, Seek};
use zip::{result::ZipResult, ZipArchive};

pub(super) fn open<R: Read + Seek>(reader: R) -> ZipResult<ZipArchive<R>> {
    let mut archive = ZipArchive::new(reader)?;
    archive.validate_header_consistency()?;
    Ok(archive)
}

pub(super) fn names<R: Read + Seek>(archive: &ZipArchive<R>) -> ZipResult<Vec<String>> {
    archive
        .file_names()
        .map(|name| name.map(|name| name.into_owned()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    #[test]
    fn rejects_conflicting_names_before_reading_model_metadata() {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for name in ["model/data.pkl", "model/data/0"] {
            writer
                .start_file(name, zip::write::SimpleFileOptions::default())
                .unwrap();
            writer.write_all(b"payload").unwrap();
        }
        let bytes = writer.finish().unwrap().into_inner();
        let mut valid = open(Cursor::new(&bytes)).unwrap();
        assert_eq!(names(&valid).unwrap(), ["model/data.pkl", "model/data/0"]);
        for index in 0..valid.len() {
            let offset = valid.by_index(index).unwrap().header_start() as usize + 30;
            let mut conflicting = bytes.clone();
            conflicting[offset] ^= 1;
            assert!(ZipArchive::new(Cursor::new(&conflicting))
                .unwrap()
                .by_index(index)
                .is_ok());
            let error = open(Cursor::new(conflicting)).err().unwrap();
            assert!(error.to_string().contains("filename mismatch"), "{error}");
        }
    }
}
