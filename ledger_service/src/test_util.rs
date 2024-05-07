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

// Macro asserting that a result is failed with a particular code and message.
#[macro_export]
macro_rules! assert_err {
    ($left:expr, $code:expr, $substr:expr) => {
        match (&$left, &$code, &$substr) {
            (left_val, code_val, substr_val) => assert!(
                (*left_val)
                    .as_ref()
                    .is_err_and(|err| err.code == *code_val && err.message.contains(*substr_val)),
                "assertion failed: \
                             `(val.err().code == code && val.err().message.contains(substr)`\n\
                             val: {:?}\n\
                             code: {:?}\n\
                             substr: {:?}",
                left_val,
                code_val,
                substr_val
            ),
        }
    };
}
