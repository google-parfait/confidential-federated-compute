// Copyright 2023 Google LLC.
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

#![no_std]
#![feature(never_type)]

extern crate alloc;

pub mod proto {
    #![allow(dead_code)]
    pub(crate) mod fcp {
        pub(crate) mod client {
            include!(concat!(env!("OUT_DIR"), "/fcp.client.rs"));
        }
        pub(crate) mod confidentialcompute {
            use prost::Message;
            include!(concat!(env!("OUT_DIR"), "/fcp.confidentialcompute.rs"));
        }
    }
    pub(crate) mod google {
        pub(crate) mod r#type {
            include!(concat!(env!("OUT_DIR"), "/google.r#type.rs"));
        }
    }

    pub use fcp::confidentialcompute::*;
}
