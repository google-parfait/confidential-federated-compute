# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
ARG debian_snapshot=sha256:f0b8edb2e4436c556493dce86b941231eead97baebb484d0d5f6ecfe4f7ed193
FROM debian@${debian_snapshot}

COPY ./main main

EXPOSE 8080

CMD ["./main"]
