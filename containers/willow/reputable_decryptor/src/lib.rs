// Copyright 2026 The Trusted Computations Platform Authors.
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

#![cfg_attr(not(feature = "std"), no_std)]
#![feature(never_type)]

extern crate alloc;
extern crate prost;
extern crate slog;
extern crate tcp_runtime;

pub mod apps {
    pub mod willow {
        pub mod reputable_decryptor {
            pub mod service {
                include!(concat!(env!("OUT_DIR"), "/apps.willow.reputable_decryptor.service.rs"));
            }
        }
    }
}

pub mod secure_aggregation {
    pub mod willow {
        include!(concat!(env!("OUT_DIR"), "/secure_aggregation.willow.rs"));
    }
}

pub mod rlwe {
    include!(concat!(env!("OUT_DIR"), "/rlwe.rs"));
}

pub mod actor;
pub mod decryptor;
