// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use alloc::vec::Vec;
use anyhow::{bail, Result};

const BLOB_ID_SIZE: usize = 16;

#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlobId {
    id: u128,
}

impl BlobId {
    pub fn from_bytes(s: &[u8]) -> Result<Self> {
        if s.len() > BLOB_ID_SIZE {
            bail!("Blob ID is longer than {} bytes", BLOB_ID_SIZE);
        }

        let mut ar = [0; BLOB_ID_SIZE];
        ar[..s.len()].copy_from_slice(s);
        Ok(BlobId { id: u128::from_le_bytes(ar) })
    }

    pub fn from_vec(v: Vec<u8>) -> Result<Self> {
        Self::from_bytes(&v)
    }

    pub fn to_vec(&self) -> Vec<u8> {
        self.id.to_le_bytes().to_vec()
    }
}

impl From<u128> for BlobId {
    fn from(id: u128) -> Self {
        BlobId { id }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use googletest::prelude::*;

    #[test]
    fn test_vec_to_blob_id() {
        assert_eq!(BlobId::from_vec(vec![]).unwrap(), BlobId::from(0));
        assert_eq!(BlobId::from_vec(vec![1]).unwrap(), BlobId::from(1));
        assert_eq!(BlobId::from_vec(vec![1; 2]).unwrap(), BlobId::from(257));
        assert_eq!(
            BlobId::from_vec(vec![1; 16]).unwrap(),
            BlobId::from(1334440654591915542993625911497130241)
        );
    }

    #[test]
    fn test_bytes_to_blob_id() {
        assert_eq!(
            BlobId::from_bytes(b"abc").unwrap(),
            BlobId::from(97 + 98 * 256 + 99 * 256 * 256)
        );
    }

    #[test]
    fn test_vec_to_blob_id_too_long() {
        assert_that!(
            BlobId::from_vec(vec![1; 17]),
            err(displays_as(contains_substring("longer than 16 bytes")))
        );
    }

    #[test]
    fn test_blob_id_to_vec() {
        assert_eq!(BlobId::from(0).to_vec(), vec![0; 16]);
        assert_eq!(
            BlobId::from(12345678).to_vec(),
            vec![78, 97, 188, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
    }
}
