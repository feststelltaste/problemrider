---
title: Fuzz-Testing
description: Testing with randomly generated input data to uncover unexpected behavior
category:
- Security
- Testing
quality_tactics_url: https://qualitytactics.de/en/security/fuzz-testing
problems:
- buffer-overflow-vulnerabilities
- inadequate-error-handling
- sql-injection-vulnerabilities
- cross-site-scripting-vulnerabilities
- null-pointer-dereferences
- integer-overflow-underflow
- legacy-code-without-tests
- stack-overflow-errors
layout: solution
---

## How to Apply ◆

> Legacy systems often contain input handling code that was never tested with unexpected, malformed, or adversarial inputs. Fuzz testing systematically generates random and semi-random inputs to discover crashes, hangs, memory errors, and other unexpected behaviors that indicate security vulnerabilities.

- Identify fuzzing targets in the legacy system: input parsers (file formats, network protocols, API payloads), deserialization routines, command-line argument processing, and any code that processes untrusted external input.
- Start with mutation-based fuzzing for legacy systems where source code is available: use tools like AFL++, libFuzzer, or Jazzer that instrument the application and mutate valid inputs to explore code paths that normal testing does not reach.
- For legacy components without source code, use black-box fuzzing tools that send randomized inputs to the application's external interfaces (HTTP endpoints, network sockets, file inputs) and monitor for crashes, error responses, and anomalous behavior.
- Implement corpus-based fuzzing by collecting real-world inputs (production request logs, sample files, protocol captures) as the initial seed corpus. Mutation-based fuzzers are more effective when starting from valid inputs that exercise meaningful code paths.
- Configure crash triage and deduplication to manage the volume of findings. Fuzzers often discover hundreds of crashes that reduce to a handful of unique root causes — automated deduplication prevents wasted investigation effort.
- Run fuzzing campaigns continuously rather than as one-time tests. Fuzzers discover deeper bugs over time as they explore more code paths, and new code changes may introduce new vulnerabilities.
- Integrate fuzz testing into CI/CD for critical input-processing components, running short fuzzing sessions (10-30 minutes) on each build to catch regressions early.

## Tradeoffs ⇄

> Fuzz testing discovers input-handling vulnerabilities that other testing methods miss, but it requires computational resources, produces results that need expert analysis, and may not be applicable to all legacy system components.

**Benefits:**

- Discovers edge-case vulnerabilities (buffer overflows, integer overflows, null pointer dereferences) that developers and conventional testing do not anticipate.
- Requires no prior knowledge of the application's expected behavior — the fuzzer discovers what makes the application fail rather than testing what should succeed.
- Can be applied to legacy code without extensive test harness construction, particularly for black-box fuzzing of external interfaces.
- Provides reproducible crash inputs that serve as both evidence of the vulnerability and regression test cases after the fix.

**Costs and Risks:**

- Fuzzing requires significant computational resources when run for extended periods, particularly for coverage-guided fuzzers that instrument the application.
- Crash triage requires security expertise to determine which crashes are exploitable vulnerabilities and which are benign failures.
- Black-box fuzzing of legacy applications can cause instability, data corruption, or resource exhaustion in the target system, requiring isolated test environments.
- Some vulnerability classes (logic errors, authorization bypasses, business logic flaws) are not discoverable through input fuzzing.

## How It Could Be

> The following scenarios illustrate how fuzz testing uncovers vulnerabilities in legacy systems.

A legacy file processing system accepts XML files uploaded by business partners for order processing. The XML parser, written in C and unchanged for 15 years, has never been tested with malformed input. The team runs AFL++ against the parser using a corpus of 200 real XML order files as seeds. After 48 hours of fuzzing, the tool discovers 7 unique crashes: 3 buffer overflows triggered by oversized element names, 2 null pointer dereferences from malformed namespace declarations, 1 integer overflow in the element depth counter, and 1 stack overflow from deeply nested elements. Two of the buffer overflows are confirmed to be exploitable for remote code execution. The team fixes all seven issues and adds the crashing inputs as permanent regression tests. They also implement a memory-safe XML parsing library to replace the custom parser.

A legacy network service processes binary protocol messages from industrial controllers. The protocol specification is partially documented, and the parsing code contains many assumptions about message structure that are not validated. The team constructs a simple fuzzing harness that sends randomized binary data to the service's network port and monitors for crashes and hangs. Within 6 hours, the fuzzer discovers that a message with a length field of zero causes the parser to enter an infinite loop, consuming 100% CPU. Another malformed message causes a heap buffer overflow when the stated payload length exceeds the actual message size. Both issues could be exploited by any device on the industrial network to crash or compromise the service. The team adds explicit length validation and input bounds checking, fixing vulnerabilities that had existed since the protocol handler was written 12 years ago.
