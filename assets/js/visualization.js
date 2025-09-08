const graph = {
  "nodes": [
    {
      "id": "abi-compatibility-issues.md",
      "title": "ABI Compatibility Issues",
      "description": "Application Binary Interface incompatibilities between different versions of libraries or system components cause runtime failures or undefined behavior.",
      "category": "Code",
      "size": 14
    },
    {
      "id": "accumulated-decision-debt.md",
      "title": "Accumulated Decision Debt",
      "description": "Deferred decisions create compound complexity for future choices, making the system increasingly difficult to evolve.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "accumulation-of-workarounds.md",
      "title": "Accumulation of Workarounds",
      "description": "Instead of fixing core issues, developers create elaborate workarounds that add complexity and technical debt to the system.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "algorithmic-complexity-problems.md",
      "title": "Algorithmic Complexity Problems",
      "description": "Code uses inefficient algorithms or data structures, leading to performance bottlenecks and resource waste.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "alignment-and-padding-issues.md",
      "title": "Alignment and Padding Issues",
      "description": "Data structures have inefficient memory layout due to poor alignment and excessive padding, wasting memory and reducing cache efficiency.",
      "category": "Architecture",
      "size": 12
    },
    {
      "id": "analysis-paralysis.md",
      "title": "Analysis Paralysis",
      "description": "Teams become stuck in research phases without moving to implementation, preventing actual progress on development work.",
      "category": "Management",
      "size": 10
    },
    {
      "id": "api-versioning-conflicts.md",
      "title": "API Versioning Conflicts",
      "description": "Inconsistent or poorly managed API versioning creates compatibility issues, breaking changes, and integration failures between services.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "approval-dependencies.md",
      "title": "Approval Dependencies",
      "description": "Work progress is frequently blocked by the need for approvals from specific individuals, creating bottlenecks and delays.",
      "category": "Dependencies",
      "size": 20
    },
    {
      "id": "architectural-mismatch.md",
      "title": "Architectural Mismatch",
      "description": "New business requirements don't fit well within existing architectural constraints, requiring extensive workarounds or compromises.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "assumption-based-development.md",
      "title": "Assumption-Based Development",
      "description": "Developers make decisions based on assumptions about requirements or user needs rather than validating their understanding.",
      "category": "Communication",
      "size": 10
    },
    {
      "id": "atomic-operation-overhead.md",
      "title": "Atomic Operation Overhead",
      "description": "Excessive use of atomic operations creates performance bottlenecks due to memory synchronization overhead and cache coherency traffic.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "authentication-bypass-vulnerabilities.md",
      "title": "Authentication Bypass Vulnerabilities",
      "description": "Security flaws that allow attackers to bypass authentication mechanisms and gain unauthorized access to protected resources.",
      "category": "Code",
      "size": 12
    },
    {
      "id": "author-frustration.md",
      "title": "Author Frustration",
      "description": "Developers become frustrated with unpredictable, conflicting, or seemingly arbitrary feedback during the code review process.",
      "category": "Culture",
      "size": 18
    },
    {
      "id": "authorization-flaws.md",
      "title": "Authorization Flaws",
      "description": "Inadequate access control mechanisms allow users to perform actions or access resources beyond their intended permissions.",
      "category": "Code",
      "size": 12
    },
    {
      "id": "automated-tooling-ineffectiveness.md",
      "title": "Automated Tooling Ineffectiveness",
      "description": "A situation where automated tooling, such as linters and formatters, is not effective because of the inconsistency of the codebase.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "avoidance-behaviors.md",
      "title": "Avoidance Behaviors",
      "description": "Complex tasks are postponed or avoided entirely due to cognitive overload, fear, or perceived difficulty.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "bikeshedding.md",
      "title": "Bikeshedding",
      "description": "Reviewers focus on trivial issues like whitespace and variable names instead of more important issues like logic and design.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "blame-culture.md",
      "title": "Blame Culture",
      "description": "Mistakes are punished instead of addressed constructively, discouraging risk-taking and learning",
      "category": "Management",
      "size": 10
    },
    {
      "id": "bloated-class.md",
      "title": "Bloated Class",
      "description": "A class that has grown so large that it has become difficult to understand, maintain, and test.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "bottleneck-formation.md",
      "title": "Bottleneck Formation",
      "description": "Specific team members, processes, or system components become constraints that limit the overall flow and productivity of development work.",
      "category": "Performance",
      "size": 20
    },
    {
      "id": "breaking-changes.md",
      "title": "Breaking Changes",
      "description": "API updates break existing client integrations, causing compatibility issues and forcing costly emergency fixes.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "brittle-codebase.md",
      "title": "Brittle Codebase",
      "description": "The existing code is difficult to modify without introducing new bugs, making maintenance and feature development risky.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "budget-overruns.md",
      "title": "Budget Overruns",
      "description": "The project costs more than originally planned due to the extra work being done, which can lead to a loss of funding and a great deal of frustration for the stakeholders.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "buffer-overflow-vulnerabilities.md",
      "title": "Buffer Overflow Vulnerabilities",
      "description": "Programs write data beyond the boundaries of allocated memory buffers, leading to security vulnerabilities and system instability.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "cache-invalidation-problems.md",
      "title": "Cache Invalidation Problems",
      "description": "Cached data becomes stale or inconsistent with the underlying data source, leading to incorrect application behavior and user confusion.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "capacity-mismatch.md",
      "title": "Capacity Mismatch",
      "description": "Available capacity at different stages of the development process doesn't match demand patterns, creating bottlenecks and underutilization.",
      "category": "Performance",
      "size": 12
    },
    {
      "id": "cargo-culting.md",
      "title": "Cargo Culting",
      "description": "Uncritical adoption of technical solutions without understanding their underlying principles and context",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "cascade-delays.md",
      "title": "Cascade Delays",
      "description": "Missed deadlines in one area cause delays in dependent work streams, creating a ripple effect that affects multiple projects and teams.",
      "category": "Business",
      "size": 18
    },
    {
      "id": "cascade-failures.md",
      "title": "Cascade Failures",
      "description": "A single change triggers a chain reaction of failures across multiple system components.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "change-management-chaos.md",
      "title": "Change Management Chaos",
      "description": "Changes to systems occur without coordination, oversight, or impact assessment, leading to conflicts and unintended consequences.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "changing-project-scope.md",
      "title": "Changing Project Scope",
      "description": "Frequent shifts in project direction confuse the team and prevent steady progress toward completion.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "circular-dependency-problems.md",
      "title": "Circular Dependency Problems",
      "description": "Components depend on each other in circular patterns, creating initialization issues, testing difficulties, and architectural complexity.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "circular-references.md",
      "title": "Circular References",
      "description": "Two or more objects reference each other in a way that prevents garbage collection, leading to memory leaks and resource exhaustion.",
      "category": "Code",
      "size": 12
    },
    {
      "id": "clever-code.md",
      "title": "Clever Code",
      "description": "Code written to demonstrate technical prowess rather than clarity, making it difficult for others to understand and maintain.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "code-duplication.md",
      "title": "Code Duplication",
      "description": "Similar or identical code exists in multiple places, making maintenance difficult and introducing inconsistency risks.",
      "category": "Architecture",
      "size": 14
    },
    {
      "id": "code-review-inefficiency.md",
      "title": "Code Review Inefficiency",
      "description": "The code review process takes excessive time, provides limited value, or creates bottlenecks in the development workflow.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "cognitive-overload.md",
      "title": "Cognitive Overload",
      "description": "Developers must maintain too many complex systems or concepts in their working memory simultaneously, reducing their effectiveness.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "communication-breakdown.md",
      "title": "Communication Breakdown",
      "description": "Team members fail to effectively share information, coordinate work, or collaborate, leading to duplicated effort and misaligned solutions.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "communication-risk-outside-project.md",
      "title": "Communication Risk Outside Project",
      "description": "External stakeholders are left uninformed, leading to surprises and misaligned expectations about project progress.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "communication-risk-within-project.md",
      "title": "Communication Risk Within Project",
      "description": "Misunderstandings and unclear messages reduce coordination and trust among project team members.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "competing-priorities.md",
      "title": "Competing Priorities",
      "description": "Multiple urgent projects or initiatives compete for the same limited resources, creating conflicts and inefficiencies.",
      "category": "Management",
      "size": 16
    },
    {
      "id": "competitive-disadvantage.md",
      "title": "Competitive Disadvantage",
      "description": "Users switch to competitors who offer better experience or more features due to technical shortcomings.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "complex-and-obscure-logic.md",
      "title": "Complex and Obscure Logic",
      "description": "The code is hard to understand due to convoluted logic, lack of comments, or poor naming conventions.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "complex-deployment-process.md",
      "title": "Complex Deployment Process",
      "description": "The process of deploying software to production is manual, time-consuming, and error-prone, which contributes to long release cycles and a high risk of failure.",
      "category": "Operations",
      "size": 20
    },
    {
      "id": "complex-domain-model.md",
      "title": "Complex Domain Model",
      "description": "The business domain being modeled in software is inherently complex, making the system difficult to understand and implement correctly.",
      "category": "Architecture",
      "size": 10
    },
    {
      "id": "complex-implementation-paths.md",
      "title": "Complex Implementation Paths",
      "description": "Simple business requirements require complex technical solutions due to architectural constraints or design limitations.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "configuration-chaos.md",
      "title": "Configuration Chaos",
      "description": "System configurations are inconsistent, difficult to manage, and prone to drift, causing unpredictable behavior across environments.",
      "category": "Operations",
      "size": 20
    },
    {
      "id": "configuration-drift.md",
      "title": "Configuration Drift",
      "description": "System configurations gradually diverge from intended standards over time, creating inconsistencies and reliability issues.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "conflicting-reviewer-opinions.md",
      "title": "Conflicting Reviewer Opinions",
      "description": "Multiple reviewers provide contradictory guidance on the same code changes, creating confusion and inefficiency.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "constant-firefighting.md",
      "title": "Constant Firefighting",
      "description": "The development team is perpetually occupied with fixing bugs and addressing urgent issues, leaving little to no time for new feature development.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "constantly-shifting-deadlines.md",
      "title": "Constantly Shifting Deadlines",
      "description": "The project's end date is repeatedly pushed back to accommodate new feature requests, leading to a loss of credibility and a great deal of frustration for the development team.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "context-switching-overhead.md",
      "title": "Context Switching Overhead",
      "description": "Developers must constantly switch between different tools, systems, or problem domains, reducing productivity and increasing cognitive load.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "convenience-driven-development.md",
      "title": "Convenience-Driven Development",
      "description": "A development practice where developers choose the easiest and most convenient solution, rather than the best solution.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "copy-paste-programming.md",
      "title": "Copy-Paste Programming",
      "description": "Developers frequently copy and paste code rather than creating reusable components, leading to maintenance nightmares and subtle bugs.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "cross-site-scripting-vulnerabilities.md",
      "title": "Cross-Site Scripting Vulnerabilities",
      "description": "Inadequate input validation and output encoding allows attackers to inject malicious scripts that execute in users' browsers.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "cross-system-data-synchronization-problems.md",
      "title": "Cross-System Data Synchronization Problems",
      "description": "Maintaining data consistency between legacy and modern systems during migration creates complex synchronization challenges and potential data corruption",
      "category": "Code",
      "size": 10
    },
    {
      "id": "customer-dissatisfaction.md",
      "title": "Customer Dissatisfaction",
      "description": "Users become frustrated with system reliability, performance, or usability issues, leading to complaints and potential customer loss.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "cv-driven-development.md",
      "title": "CV Driven Development",
      "description": "Choosing technologies or practices primarily to enhance personal resumes rather than solve business problems",
      "category": "Code",
      "size": 10
    },
    {
      "id": "data-migration-complexities.md",
      "title": "Data Migration Complexities",
      "description": "Complex data migration processes create risks of data loss, corruption, or extended downtime during system updates.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "data-migration-integrity-issues.md",
      "title": "Data Migration Integrity Issues",
      "description": "Data loses integrity, consistency, or meaning during migration from legacy to modern systems due to schema mismatches and format incompatibilities",
      "category": "Code",
      "size": 14
    },
    {
      "id": "data-protection-risk.md",
      "title": "Data Protection Risk",
      "description": "Handling of personal or sensitive data lacks safeguards, exposing the project to legal and ethical issues",
      "category": "Process",
      "size": 16
    },
    {
      "id": "data-structure-cache-inefficiency.md",
      "title": "Data Structure Cache Inefficiency",
      "description": "Data structures are organized in ways that cause poor cache performance, leading to excessive memory access latency and reduced throughput.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "database-connection-leaks.md",
      "title": "Database Connection Leaks",
      "description": "Database connections are opened but not properly closed, leading to connection pool exhaustion and application failures.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "database-query-performance-issues.md",
      "title": "Database Query Performance Issues",
      "description": "Poorly optimized database queries cause slow response times, high resource consumption, and scalability problems.",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "database-schema-design-problems.md",
      "title": "Database Schema Design Problems",
      "description": "Poor database schema design creates performance issues, data integrity problems, and maintenance difficulties.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "deadline-pressure.md",
      "title": "Deadline Pressure",
      "description": "Intense pressure to meet deadlines leads to rushed decisions, shortcuts, and compromised quality in software development.",
      "category": "Code",
      "size": 14
    },
    {
      "id": "deadlock-conditions.md",
      "title": "Deadlock Conditions",
      "description": "Multiple threads or processes wait indefinitely for each other to release resources, causing system freeze and application unresponsiveness.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "debugging-difficulties.md",
      "title": "Debugging Difficulties",
      "description": "Finding and fixing bugs becomes challenging due to complex code architecture, poor logging, or inadequate development tools.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "decision-avoidance.md",
      "title": "Decision Avoidance",
      "description": "Important technical decisions are repeatedly deferred, preventing progress and creating bottlenecks in development work.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "decision-paralysis.md",
      "title": "Decision Paralysis",
      "description": "Developers struggle to make choices without clear guidance, which can lead to a slowdown in development and a great deal of frustration for the development team.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "declining-business-metrics.md",
      "title": "Declining Business Metrics",
      "description": "Key business indicators such as user engagement, conversion rates, revenue, and stakeholder confidence deteriorate due to technical problems.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "defensive-coding-practices.md",
      "title": "Defensive Coding Practices",
      "description": "Developers write overly verbose code, excessive comments, or unnecessary defensive logic to preempt anticipated criticism during code reviews.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "delayed-bug-fixes.md",
      "title": "Delayed Bug Fixes",
      "description": "Known issues remain unresolved for extended periods, causing ongoing problems and user frustration.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "delayed-decision-making.md",
      "title": "Delayed Decision Making",
      "description": "Important decisions that affect development progress are postponed or take excessive time to make, creating bottlenecks and uncertainty.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "delayed-issue-resolution.md",
      "title": "Delayed Issue Resolution",
      "description": "Problems persist longer because no one feels responsible for fixing them, leading to accumulated technical debt and user frustration.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "delayed-project-timelines.md",
      "title": "Delayed Project Timelines",
      "description": "Projects consistently take longer than planned, missing deadlines and extending delivery schedules beyond original estimates.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "delayed-value-delivery.md",
      "title": "Delayed Value Delivery",
      "description": "Users have to wait for an extended period to receive new features or bug fixes, leading to frustration and a competitive disadvantage.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "dependency-on-supplier.md",
      "title": "Dependency on Supplier",
      "description": "External vendors control critical parts of the system, reducing organizational flexibility and increasing lock-in risk.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "dependency-version-conflicts.md",
      "title": "Dependency Version Conflicts",
      "description": "Conflicting versions of dependencies cause runtime errors, build failures, and unexpected behavior in applications.",
      "category": "Code",
      "size": 16
    },
    {
      "id": "deployment-coupling.md",
      "title": "Deployment Coupling",
      "description": "A situation where multiple components or services must be deployed together, even if only one of them has changed.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "deployment-environment-inconsistencies.md",
      "title": "Deployment Environment Inconsistencies",
      "description": "Differences between deployment environments cause applications to behave differently or fail when moved between environments.",
      "category": "Operations",
      "size": 20
    },
    {
      "id": "deployment-risk.md",
      "title": "Deployment Risk",
      "description": "System deployments carry high risk of failure or damage due to irreversible changes and lack of recovery mechanisms.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "developer-frustration-and-burnout.md",
      "title": "Developer Frustration and Burnout",
      "description": "Developers are demotivated, disengaged, and exhausted due to persistent issues in the work environment and codebase.",
      "category": "Culture",
      "size": 10
    },
    {
      "id": "development-disruption.md",
      "title": "Development Disruption",
      "description": "The development team is constantly interrupted by urgent production issues, which disrupts planned work and reduces overall productivity.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "difficult-code-comprehension.md",
      "title": "Difficult Code Comprehension",
      "description": "A situation where developers have a hard time understanding the codebase.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "difficult-code-reuse.md",
      "title": "Difficult Code Reuse",
      "description": "It is difficult to reuse code in different contexts because it is not designed in a modular and reusable way.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "difficult-developer-onboarding.md",
      "title": "Difficult Developer Onboarding",
      "description": "New team members take an unusually long time to become productive due to complex systems, poor documentation, and inadequate onboarding processes.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "difficult-to-test-code.md",
      "title": "Difficult to Test Code",
      "description": "Components cannot be easily tested in isolation due to tight coupling, global dependencies, or complex setup requirements.",
      "category": "Code",
      "size": 16
    },
    {
      "id": "difficult-to-understand-code.md",
      "title": "Difficult to Understand Code",
      "description": "It's hard to grasp the purpose of modules or functions without understanding many other parts of the system, slowing development and increasing errors.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "difficulty-quantifying-benefits.md",
      "title": "Difficulty Quantifying Benefits",
      "description": "It is hard to measure the ROI of refactoring work compared to new features, so technical improvements often lose out in prioritization decisions.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "dma-coherency-issues.md",
      "title": "DMA Coherency Issues",
      "description": "Direct Memory Access operations conflict with CPU cache coherency, leading to data corruption or inconsistent data views between CPU and DMA devices.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "duplicated-effort.md",
      "title": "Duplicated Effort",
      "description": "Multiple team members unknowingly work on the same problems or implement similar solutions independently.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "duplicated-research-effort.md",
      "title": "Duplicated Research Effort",
      "description": "Multiple team members research the same topics independently, wasting time and failing to build collective knowledge.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "duplicated-work.md",
      "title": "Duplicated Work",
      "description": "Multiple team members unknowingly work on the same tasks or solve the same problems, leading to wasted effort and potential conflicts.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "eager-to-please-stakeholders.md",
      "title": "Eager to Please Stakeholders",
      "description": "The project team agrees to every new request from the stakeholders without pushing back or explaining the trade-offs, which can lead to scope creep and a number of other problems.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "endianness-conversion-overhead.md",
      "title": "Endianness Conversion Overhead",
      "description": "Frequent byte order conversions between different endianness formats create performance overhead in data processing and network communication.",
      "category": "Code",
      "size": 11
    },
    {
      "id": "environment-variable-issues.md",
      "title": "Environment Variable Issues",
      "description": "Improper management of environment variables causes configuration problems, security vulnerabilities, and deployment failures.",
      "category": "Operations",
      "size": 20
    },
    {
      "id": "error-message-information-disclosure.md",
      "title": "Error Message Information Disclosure",
      "description": "Error messages reveal sensitive system information that can be exploited by attackers to understand system architecture and vulnerabilities.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "excessive-class-size.md",
      "title": "Excessive Class Size",
      "description": "Classes become overly large and complex, making them difficult to understand, maintain, and test.",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "excessive-disk-io.md",
      "title": "Excessive Disk I/O",
      "description": "The system performs a high number of disk read/write operations, indicating inefficient data access or processing.",
      "category": "Performance",
      "size": 20
    },
    {
      "id": "excessive-logging.md",
      "title": "Excessive Logging",
      "description": "Applications generate a very high volume of logs, consuming excessive disk space and potentially impacting performance.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "excessive-object-allocation.md",
      "title": "Excessive Object Allocation",
      "description": "Code creates a large number of temporary objects, putting pressure on the garbage collector and degrading performance.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "extended-cycle-times.md",
      "title": "Extended Cycle Times",
      "description": "The time from when work begins until it's completed and delivered becomes much longer than the actual work time required.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "extended-research-time.md",
      "title": "Extended Research Time",
      "description": "Developers spend significant portions of their day researching rather than implementing, due to knowledge gaps or complex legacy systems.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "extended-review-cycles.md",
      "title": "Extended Review Cycles",
      "description": "Code reviews require multiple rounds of feedback and revision, significantly extending the time from code submission to approval.",
      "category": "Process",
      "size": 14
    },
    {
      "id": "external-service-delays.md",
      "title": "External Service Delays",
      "description": "An API depends on other services (third-party or internal) that are slow to respond, causing the API itself to be slow.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "false-sharing.md",
      "title": "False Sharing",
      "description": "Multiple CPU cores access different variables located on the same cache line, causing unnecessary cache coherency traffic and performance degradation.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "fear-of-breaking-changes.md",
      "title": "Fear of Breaking Changes",
      "description": "The team is reluctant to make changes to the codebase for fear of breaking existing functionality, which can lead to a stagnant and outdated system.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "fear-of-change.md",
      "title": "Fear of Change",
      "description": "Developers are hesitant to modify existing code due to the high risk of breaking something.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "fear-of-conflict.md",
      "title": "Fear of Conflict",
      "description": "Reviewers avoid challenging complex logic or design decisions, opting for easier, less confrontational feedback.",
      "category": "Communication",
      "size": 12
    },
    {
      "id": "fear-of-failure.md",
      "title": "Fear of Failure",
      "description": "A pervasive fear of making mistakes or failing can lead to inaction, risk aversion, and a reluctance to innovate within a development team.",
      "category": "Culture",
      "size": 14
    },
    {
      "id": "feature-bloat.md",
      "title": "Feature Bloat",
      "description": "Products become overly complex with numerous features that dilute the core value proposition and confuse users.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "feature-creep-without-refactoring.md",
      "title": "Feature Creep Without Refactoring",
      "description": "The continuous addition of new features to a codebase without taking the time to refactor and improve the design.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "feature-creep.md",
      "title": "Feature Creep",
      "description": "The scope of a feature or component gradually expands over time, leading to a complex and bloated system that is difficult to maintain.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "feature-factory.md",
      "title": "Feature Factory",
      "description": "Organization prioritizes shipping features over understanding their business impact and user value",
      "category": "Management",
      "size": 10
    },
    {
      "id": "feature-gaps.md",
      "title": "Feature Gaps",
      "description": "Important functionality is missing because developers assumed it wasn't needed, creating incomplete solutions that don't meet user needs.",
      "category": "Business",
      "size": 16
    },
    {
      "id": "feedback-isolation.md",
      "title": "Feedback Isolation",
      "description": "Development teams operate without regular input from stakeholders and users, leading to products that miss requirements and user needs.",
      "category": "Business",
      "size": 18
    },
    {
      "id": "flaky-tests.md",
      "title": "Flaky Tests",
      "description": "Tests fail randomly due to timing, setup, or dependencies, undermining trust in the test suite",
      "category": "Code",
      "size": 10
    },
    {
      "id": "frequent-changes-to-requirements.md",
      "title": "Frequent Changes to Requirements",
      "description": "The requirements for a project or feature are constantly being updated, even after development has started, leading to rework, delays, and frustration.",
      "category": "Communication",
      "size": 10
    },
    {
      "id": "frequent-hotfixes-and-rollbacks.md",
      "title": "Frequent Hotfixes and Rollbacks",
      "description": "The team is constantly deploying small fixes or rolling back releases due to insufficient testing and quality control.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "garbage-collection-pressure.md",
      "title": "Garbage Collection Pressure",
      "description": "Excessive object allocation and deallocation causes frequent garbage collection cycles, creating performance pauses and reducing application throughput.",
      "category": "Code",
      "size": 12
    },
    {
      "id": "global-state-and-side-effects.md",
      "title": "Global State and Side Effects",
      "description": "Excessive use of global variables or functions with hidden side effects makes it difficult to reason about code behavior.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "god-object-anti-pattern.md",
      "title": "God Object Anti-Pattern",
      "description": "Single classes or components handle too many responsibilities, becoming overly complex and difficult to maintain or test.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "gold-plating.md",
      "title": "Gold Plating",
      "description": "Developers add unnecessary features or complexity to a project because they believe it will impress the stakeholders, even if it was not requested.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "gradual-performance-degradation.md",
      "title": "Gradual Performance Degradation",
      "description": "Application performance slowly deteriorates over time due to resource leaks, accumulating technical debt, or inefficient algorithms.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "graphql-complexity-issues.md",
      "title": "GraphQL Complexity Issues",
      "description": "GraphQL queries become too complex or expensive to execute, causing performance problems and potential denial-of-service vulnerabilities.",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "growing-task-queues.md",
      "title": "Growing Task Queues",
      "description": "Asynchronous processing queues accumulate unprocessed tasks, indicating a bottleneck in the processing pipeline.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "hardcoded-values.md",
      "title": "Hardcoded Values",
      "description": "Magic numbers and fixed strings reduce flexibility, making configuration and adaptation difficult",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "hidden-dependencies.md",
      "title": "Hidden Dependencies",
      "description": "Workarounds and patches create unexpected dependencies between system components that are not obvious from the code structure.",
      "category": "Architecture",
      "size": 14
    },
    {
      "id": "hidden-side-effects.md",
      "title": "Hidden Side Effects",
      "description": "Functions have undocumented side effects that modify state or trigger actions beyond their apparent purpose.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "high-api-latency.md",
      "title": "High API Latency",
      "description": "The time it takes for an API to respond to a request is excessively long, leading to poor application performance and a negative user experience.",
      "category": "Performance",
      "size": 20
    },
    {
      "id": "high-bug-introduction-rate.md",
      "title": "High Bug Introduction Rate",
      "description": "A high rate of new bugs are introduced with every change to the codebase, indicating underlying quality issues.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "high-client-side-resource-consumption.md",
      "title": "High Client-Side Resource Consumption",
      "description": "Client applications consume excessive CPU or memory, leading to sluggish performance and poor user experience.",
      "category": "Performance",
      "size": 20
    },
    {
      "id": "high-connection-count.md",
      "title": "High Connection Count",
      "description": "A large number of open database connections, even if idle, can consume significant memory resources and lead to connection rejections.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "high-coupling-low-cohesion.md",
      "title": "High Coupling and Low Cohesion",
      "description": "Software components are overly dependent on each other and perform too many unrelated functions, making the system difficult to change and understand.",
      "category": "Architecture",
      "size": 11
    },
    {
      "id": "high-database-resource-utilization.md",
      "title": "High Database Resource Utilization",
      "description": "The database server consistently operates with high CPU or memory usage, risking instability and slowing down all dependent services.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "high-defect-rate-in-production.md",
      "title": "High Defect Rate in Production",
      "description": "A large number of bugs are discovered in the live environment after a release, indicating underlying issues in the development and quality assurance process.",
      "category": "Business",
      "size": 10
    },
    {
      "id": "high-maintenance-costs.md",
      "title": "High Maintenance Costs",
      "description": "A disproportionately large amount of the development budget and effort is consumed by maintaining the existing system rather than creating new value.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "high-number-of-database-queries.md",
      "title": "High Number of Database Queries",
      "description": "A single user request triggers an unexpectedly large number of database queries, leading to performance degradation and increased database load.",
      "category": "Data",
      "size": 20
    },
    {
      "id": "high-resource-utilization-on-client.md",
      "title": "High Resource Utilization on Client",
      "description": "Client applications may consume excessive CPU or memory, leading to a poor user experience, especially on less powerful devices.",
      "category": "Performance",
      "size": 20
    },
    {
      "id": "high-technical-debt.md",
      "title": "High Technical Debt",
      "description": "Accumulation of design or implementation shortcuts that lead to increased costs and effort in the long run.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "high-turnover.md",
      "title": "High Turnover",
      "description": "New developers become frustrated and leave the team due to poor onboarding and system complexity.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "history-of-failed-changes.md",
      "title": "History of Failed Changes",
      "description": "A past record of failed deployments or changes creates a culture of fear and resistance to future modifications.",
      "category": "Culture",
      "size": 20
    },
    {
      "id": "immature-delivery-strategy.md",
      "title": "Immature Delivery Strategy",
      "description": "Software rollout processes are improvised, inconsistent, or inadequately planned, increasing downtime and user confusion.",
      "category": "Operations",
      "size": 20
    },
    {
      "id": "imperative-data-fetching-logic.md",
      "title": "Imperative Data Fetching Logic",
      "description": "The application code is written in a way that fetches data in a loop, rather than using a more efficient, declarative approach, leading to performance problems.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "implementation-rework.md",
      "title": "Implementation Rework",
      "description": "Features must be rebuilt when initial understanding proves incorrect, wasting development effort and delaying delivery.",
      "category": "Code",
      "size": 14
    },
    {
      "id": "implementation-starts-without-design.md",
      "title": "Implementation Starts Without Design",
      "description": "Development begins with unclear structure, leading to disorganized code and architectural drift",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "implicit-knowledge.md",
      "title": "Implicit Knowledge",
      "description": "Critical system knowledge exists as unwritten assumptions, tribal knowledge, and undocumented practices rather than being explicitly captured.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "improper-event-listener-management.md",
      "title": "Improper Event Listener Management",
      "description": "Event listeners are added but not removed when associated objects are destroyed, creating memory leaks and preventing garbage collection.",
      "category": "Architecture",
      "size": 14
    },
    {
      "id": "inability-to-innovate.md",
      "title": "Inability to Innovate",
      "description": "The team is so bogged down in day-to-day maintenance tasks that they have no time to think about future improvements or new approaches.",
      "category": "Management",
      "size": 12
    },
    {
      "id": "inadequate-code-reviews.md",
      "title": "Review Process Breakdown",
      "description": "Code reviews are not consistently performed, are rushed, superficial, or fail to identify critical issues, leading to lower code quality and increased risk.",
      "category": "Code",
      "size": 12
    },
    {
      "id": "inadequate-configuration-management.md",
      "title": "Inadequate Configuration Management",
      "description": "Versions of code, data, or infrastructure are not tracked properly, leading to errors or rollback issues",
      "category": "Code",
      "size": 14
    },
    {
      "id": "inadequate-error-handling.md",
      "title": "Inadequate Error Handling",
      "description": "Poor error handling mechanisms fail to gracefully manage exceptions, leading to application crashes and poor user experiences.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "inadequate-initial-reviews.md",
      "title": "Inadequate Initial Reviews",
      "description": "First-round code reviews are incomplete or superficial, failing to identify important issues that are discovered in later review rounds.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "inadequate-integration-tests.md",
      "title": "Inadequate Integration Tests",
      "description": "The interactions between different modules or services are not thoroughly tested, leading to integration failures.",
      "category": "Architecture",
      "size": 16
    },
    {
      "id": "inadequate-mentoring-structure.md",
      "title": "Inadequate Mentoring Structure",
      "description": "The organization lacks a systematic approach to mentoring new developers, leading to inconsistent guidance and support.",
      "category": "Communication",
      "size": 10
    },
    {
      "id": "inadequate-onboarding.md",
      "title": "Inadequate Onboarding",
      "description": "New users struggle to understand how to use the application, leading to early abandonment.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "inadequate-requirements-gathering.md",
      "title": "Inadequate Requirements Gathering",
      "description": "Insufficient analysis and documentation of requirements leads to building solutions that don't meet actual needs.",
      "category": "Process",
      "size": 18
    },
    {
      "id": "inadequate-test-data-management.md",
      "title": "Inadequate Test Data Management",
      "description": "The use of unrealistic, outdated, or insufficient test data leads to tests that do not accurately reflect real-world scenarios.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "inadequate-test-infrastructure.md",
      "title": "Inadequate Test Infrastructure",
      "description": "Missing tools, environments, or automation make thorough testing slow or impossible",
      "category": "Code",
      "size": 16
    },
    {
      "id": "inappropriate-skillset.md",
      "title": "Inappropriate Skillset",
      "description": "Team members lack essential knowledge or experience needed for their assigned roles and responsibilities.",
      "category": "Culture",
      "size": 20
    },
    {
      "id": "incomplete-knowledge.md",
      "title": "Incomplete Knowledge",
      "description": "Developers are unaware of all the locations where similar logic exists, which can lead to synchronization problems and other issues.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "incomplete-projects.md",
      "title": "Incomplete Projects",
      "description": "Features are started but never finished due to shifting priorities, leading to a great deal of wasted effort and a sense of frustration for the development team.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "inconsistent-behavior.md",
      "title": "Inconsistent Behavior",
      "description": "The same business process produces different outcomes depending on where it's triggered, leading to a confusing and unpredictable user experience.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "inconsistent-codebase.md",
      "title": "Inconsistent Codebase",
      "description": "The project's code lacks uniform style, coding standards, and design patterns, making it difficult to read, maintain, and onboard new developers.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "inconsistent-coding-standards.md",
      "title": "Inconsistent Coding Standards",
      "description": "Lack of uniform coding standards across the codebase creates maintenance difficulties and reduces code readability and quality.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "inconsistent-execution.md",
      "title": "Inconsistent Execution",
      "description": "Manual processes lead to variations in how tasks are completed across team members and over time, creating unpredictable outcomes.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "inconsistent-knowledge-acquisition.md",
      "title": "Inconsistent Knowledge Acquisition",
      "description": "New team members learn different aspects and depths of system knowledge depending on their mentor or learning path, creating uneven skill distribution.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "inconsistent-naming-conventions.md",
      "title": "Inconsistent Naming Conventions",
      "description": "Unstructured or conflicting names make code harder to read, navigate, and maintain",
      "category": "Code",
      "size": 10
    },
    {
      "id": "inconsistent-onboarding-experience.md",
      "title": "Inconsistent Onboarding Experience",
      "description": "New team members receive dramatically different onboarding experiences depending on who is available to help them, creating unequal outcomes.",
      "category": "Communication",
      "size": 10
    },
    {
      "id": "inconsistent-quality.md",
      "title": "Inconsistent Quality",
      "description": "Some parts of the system are well-maintained while others deteriorate, creating unpredictable user experiences and maintenance challenges.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "incorrect-index-type.md",
      "title": "Incorrect Index Type",
      "description": "Using an inappropriate type of database index for a given query pattern, leading to inefficient data retrieval.",
      "category": "Performance",
      "size": 15
    },
    {
      "id": "incorrect-max-connection-pool-size.md",
      "title": "Incorrect Max Connection Pool Size",
      "description": "The maximum number of connections in a database connection pool is set incorrectly, leading to either wasted resources or connection exhaustion.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "increased-bug-count.md",
      "title": "Increased Bug Count",
      "description": "Changes introduce new defects more frequently, leading to a higher defect rate in production and degraded software quality.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "increased-cognitive-load.md",
      "title": "Increased Cognitive Load",
      "description": "Developers must expend excessive mental energy to understand and work with inconsistent, complex, or poorly structured code.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "increased-cost-of-development.md",
      "title": "Increased Cost of Development",
      "description": "The cost of fixing bugs and maintaining poor code is significantly higher than preventing issues initially.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "increased-customer-support-load.md",
      "title": "Increased Customer Support Load",
      "description": "Users contact support more frequently due to frustration or inability to complete tasks.",
      "category": "Business",
      "size": 10
    },
    {
      "id": "increased-error-rates.md",
      "title": "Increased Error Rates",
      "description": "An unusual or sustained rise in the frequency of errors reported by an application or service.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "increased-manual-testing-effort.md",
      "title": "Increased Manual Testing Effort",
      "description": "A disproportionate amount of time is spent on manual testing due to a lack of automation.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "increased-manual-work.md",
      "title": "Increased Manual Work",
      "description": "Developers spend time on repetitive tasks that should be automated, reducing time available for actual development work.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "increased-risk-of-bugs.md",
      "title": "Increased Risk of Bugs",
      "description": "Code complexity and lack of clarity make it more likely that developers will introduce defects when making changes.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "increased-stress-and-burnout.md",
      "title": "Increased Stress and Burnout",
      "description": "Team members are overworked and stressed due to unrealistic expectations and time pressure, leading to decreased morale and productivity.",
      "category": "Culture",
      "size": 20
    },
    {
      "id": "increased-technical-shortcuts.md",
      "title": "Increased Technical Shortcuts",
      "description": "Pressure to deliver leads to more quick fixes and workarounds instead of proper solutions, creating future maintenance problems.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "increased-time-to-market.md",
      "title": "Increased Time to Market",
      "description": "It takes longer to get new features and products to market, potentially resulting in loss of competitive advantage and revenue opportunities.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "increasing-brittleness.md",
      "title": "Increasing Brittleness",
      "description": "Software systems become more fragile and prone to breaking over time, with small changes having unpredictable and widespread effects.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "index-fragmentation.md",
      "title": "Index Fragmentation",
      "description": "Over time, as data is inserted, updated, and deleted, database indexes become disorganized, reducing their efficiency.",
      "category": "Data",
      "size": 20
    },
    {
      "id": "individual-recognition-culture.md",
      "title": "Individual Recognition Culture",
      "description": "A culture where individual accomplishments are valued more than team accomplishments.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "inefficient-code.md",
      "title": "Inefficient Code",
      "description": "The code responsible for handling a request is computationally expensive or contains performance bottlenecks.",
      "category": "Performance",
      "size": 12
    },
    {
      "id": "inefficient-database-indexing.md",
      "title": "Inefficient Database Indexing",
      "description": "The database lacks appropriate indexes for common query patterns, forcing slow, full-table scans for data retrieval operations.",
      "category": "Data",
      "size": 14
    },
    {
      "id": "inefficient-development-environment.md",
      "title": "Inefficient Development Environment",
      "description": "The team is slowed down by a slow and cumbersome development environment",
      "category": "Code",
      "size": 10
    },
    {
      "id": "inefficient-frontend-code.md",
      "title": "Inefficient Frontend Code",
      "description": "Unoptimized JavaScript, excessive DOM manipulation, or complex CSS animations that are computationally expensive.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "inefficient-processes.md",
      "title": "Inefficient Processes",
      "description": "Poor workflows, excessive meetings, or bureaucratic procedures waste development time and reduce team productivity.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "inexperienced-developers.md",
      "title": "Inexperienced Developers",
      "description": "Development team lacks the knowledge and experience to implement best practices and maintainable solutions.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "information-decay.md",
      "title": "Information Decay",
      "description": "System documentation becomes outdated, inaccurate, or incomplete over time, making it unreliable for decision-making and system understanding.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "information-fragmentation.md",
      "title": "Information Fragmentation",
      "description": "Critical system knowledge is scattered across multiple locations and formats, making it difficult to find and maintain.",
      "category": "Communication",
      "size": 18
    },
    {
      "id": "insecure-data-transmission.md",
      "title": "Insecure Data Transmission",
      "description": "Sensitive data transmitted without proper encryption or security controls, exposing it to interception and unauthorized access.",
      "category": "Performance",
      "size": 10
    },
    {
      "id": "insufficient-audit-logging.md",
      "title": "Insufficient Audit Logging",
      "description": "Inadequate logging of security-relevant events makes it difficult to detect breaches, investigate incidents, or maintain compliance.",
      "category": "Code",
      "size": 12
    },
    {
      "id": "insufficient-code-review.md",
      "title": "Review Process Breakdown",
      "description": "Code review processes fail to catch design flaws, bugs, or quality issues due to inadequate depth, time, or expertise.",
      "category": "Code",
      "size": 12
    },
    {
      "id": "insufficient-design-skills.md",
      "title": "Insufficient Design Skills",
      "description": "The development team lacks the necessary skills and experience to design and build well-structured, maintainable software.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "insufficient-testing.md",
      "title": "Quality Blind Spots",
      "description": "The testing process is not comprehensive enough, leading to a high defect rate in production.",
      "category": "Code",
      "size": 16
    },
    {
      "id": "insufficient-worker-capacity.md",
      "title": "Insufficient Worker Capacity",
      "description": "There are not enough worker processes or threads to handle the incoming volume of tasks in an asynchronous system, leading to growing queues.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "integer-overflow-underflow.md",
      "title": "Integer Overflow and Underflow",
      "description": "Arithmetic operations produce results that exceed the maximum or minimum values representable by integer data types, leading to unexpected behavior.",
      "category": "Code",
      "size": 11
    },
    {
      "id": "integration-difficulties.md",
      "title": "Integration Difficulties",
      "description": "Connecting with modern services requires extensive workarounds due to architectural limitations or outdated integration patterns.",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "interrupt-overhead.md",
      "title": "Interrupt Overhead",
      "description": "Excessive hardware interrupts disrupt CPU execution flow, causing frequent context switches and reducing application performance.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "invisible-nature-of-technical-debt.md",
      "title": "Invisible Nature of Technical Debt",
      "description": "The impact of technical debt is not visible to non-technical stakeholders, making it hard to justify addressing it and allocate resources for improvement.",
      "category": "Communication",
      "size": 14
    },
    {
      "id": "knowledge-dependency.md",
      "title": "Knowledge Dependency",
      "description": "Team members remain dependent on specific experienced individuals for knowledge and decision-making longer than appropriate for their role and tenure.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "knowledge-gaps.md",
      "title": "Knowledge Gaps",
      "description": "Lack of understanding about systems, business requirements, or technical domains leads to extended research time and suboptimal solutions.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "knowledge-sharing-breakdown.md",
      "title": "Knowledge Sharing Breakdown",
      "description": "The process of sharing knowledge and expertise among team members is ineffective, leading to information silos and reduced team learning.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "knowledge-silos.md",
      "title": "Knowledge Silos",
      "description": "Important research findings and expertise remain isolated to individual team members, preventing knowledge sharing and team learning.",
      "category": "Culture",
      "size": 16
    },
    {
      "id": "lack-of-ownership-and-accountability.md",
      "title": "Lack of Ownership and Accountability",
      "description": "No clear responsibility for maintaining code quality, documentation, or specific system components over time.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "language-barriers.md",
      "title": "Language Barriers",
      "description": "Differences in language or terminology hinder smooth communication and understanding within the team.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "large-estimates-for-small-changes.md",
      "title": "Large Estimates for Small Changes",
      "description": "The team consistently provides large time estimates for seemingly small changes, indicating underlying code complexity and risk.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "large-feature-scope.md",
      "title": "Large Feature Scope",
      "description": "Features are too large to be broken down into smaller, incremental changes, leading to long-lived branches and integration problems.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "large-pull-requests.md",
      "title": "Large Pull Requests",
      "description": "Pull requests are too large to review effectively, leading to superficial reviews and missed issues.",
      "category": "Code",
      "size": 11
    },
    {
      "id": "large-risky-releases.md",
      "title": "Large, Risky Releases",
      "description": "Infrequent releases lead to large, complex deployments that are difficult to test, prone to failure, and have a significant impact on users when they go wrong.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "lazy-loading.md",
      "title": "Lazy Loading",
      "description": "The use of lazy loading in an ORM framework leads to a large number of unnecessary database queries, which can significantly degrade application performance.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "legacy-api-versioning-nightmare.md",
      "title": "Legacy API Versioning Nightmare",
      "description": "Legacy systems with poorly designed APIs create versioning and backward compatibility challenges that compound over time",
      "category": "Architecture",
      "size": 10
    },
    {
      "id": "legacy-business-logic-extraction-difficulty.md",
      "title": "Legacy Business Logic Extraction Difficulty",
      "description": "Critical business rules are embedded deep within legacy code structures, making them nearly impossible to identify and extract",
      "category": "Architecture",
      "size": 12
    },
    {
      "id": "legacy-code-without-tests.md",
      "title": "Legacy Code Without Tests",
      "description": "Existing legacy systems often lack automated tests, making it challenging to add them incrementally and safely modify the code.",
      "category": "Code",
      "size": 12
    },
    {
      "id": "legacy-configuration-management-chaos.md",
      "title": "Legacy Configuration Management Chaos",
      "description": "Configuration settings are hardcoded, undocumented, or stored in proprietary formats that prevent modern deployment practices",
      "category": "Code",
      "size": 20
    },
    {
      "id": "legacy-skill-shortage.md",
      "title": "Legacy Skill Shortage",
      "description": "Critical shortage of developers with knowledge of legacy technologies creates bottlenecks and single points of failure for system maintenance",
      "category": "Management",
      "size": 10
    },
    {
      "id": "legacy-system-documentation-archaeology.md",
      "title": "Legacy System Documentation Archaeology",
      "description": "Critical system knowledge exists only in obsolete documentation formats, outdated diagrams, and departed employees' tribal knowledge",
      "category": "Communication",
      "size": 12
    },
    {
      "id": "legal-disputes.md",
      "title": "Legal Disputes",
      "description": "Conflicts over contracts, deliverables, or responsibilities escalate to legal proceedings, consuming resources and damaging relationships.",
      "category": "Dependencies",
      "size": 20
    },
    {
      "id": "limited-team-learning.md",
      "title": "Limited Team Learning",
      "description": "A situation where a team does not learn from its mistakes and does not improve over time.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "load-balancing-problems.md",
      "title": "Load Balancing Problems",
      "description": "Load balancing mechanisms distribute traffic inefficiently or fail to adapt to changing conditions, causing performance issues and service instability.",
      "category": "Operations",
      "size": 20
    },
    {
      "id": "lock-contention.md",
      "title": "Lock Contention",
      "description": "Multiple threads compete for the same locks, causing threads to block and reducing parallel execution efficiency.",
      "category": "Code",
      "size": 15
    },
    {
      "id": "log-injection-vulnerabilities.md",
      "title": "Log Injection Vulnerabilities",
      "description": "Unsanitized user input in log messages allows attackers to inject malicious content that can compromise log integrity or exploit log processing systems.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "log-spam.md",
      "title": "Log Spam",
      "description": "The application or database logs are flooded with a large number of similar-looking queries, making it difficult to identify and diagnose other issues.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "logging-configuration-issues.md",
      "title": "Logging Configuration Issues",
      "description": "Improper logging configuration results in missing critical information, excessive log volume, or security vulnerabilities.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "long-build-and-test-times.md",
      "title": "Long Build and Test Times",
      "description": "A situation where it takes a long time to build and test a system.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "long-lived-feature-branches.md",
      "title": "Long-Lived Feature Branches",
      "description": "Code is not being reviewed and merged in a timely manner, leading to integration problems and increased risk.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "long-release-cycles.md",
      "title": "Long Release Cycles",
      "description": "Releases are delayed due to prolonged manual testing phases or last-minute bug discoveries.",
      "category": "Management",
      "size": 14
    },
    {
      "id": "long-running-database-transactions.md",
      "title": "Long-Running Database Transactions",
      "description": "Database transactions remain open for extended periods, holding locks and consuming resources, which can block other operations.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "long-running-transactions.md",
      "title": "Long-Running Transactions",
      "description": "Database transactions that remain open for a long time can hold locks, consume resources, and block other operations.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "lower-code-quality.md",
      "title": "Lower Code Quality",
      "description": "Burned-out or rushed developers are more likely to make mistakes, leading to an increase in defects.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "maintenance-bottlenecks.md",
      "title": "Maintenance Bottlenecks",
      "description": "A situation where a small number of developers are the only ones who can make changes to a critical part of the system.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "maintenance-cost-increase.md",
      "title": "Maintenance Cost Increase",
      "description": "The resources required to maintain, support, and update software systems grow over time, consuming increasing portions of development budgets.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "maintenance-overhead.md",
      "title": "Maintenance Overhead",
      "description": "A disproportionate amount of time and effort is spent on maintaining the existing system, often due to duplicated code and a lack of reusable components.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "maintenance-paralysis.md",
      "title": "Maintenance Paralysis",
      "description": "Teams avoid necessary improvements because they cannot verify that changes don't break existing functionality.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "manual-deployment-processes.md",
      "title": "Manual Deployment Processes",
      "description": "Releases require human intervention, increasing the chance for mistakes and inconsistencies",
      "category": "Code",
      "size": 20
    },
    {
      "id": "market-pressure.md",
      "title": "Market Pressure",
      "description": "External competitive forces or market conditions drive rushed decisions, scope changes, and unrealistic expectations.",
      "category": "Business",
      "size": 14
    },
    {
      "id": "memory-barrier-inefficiency.md",
      "title": "Memory Barrier Inefficiency",
      "description": "Excessive or incorrectly placed memory barriers disrupt CPU pipeline optimization and reduce performance in multi-threaded applications.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "memory-fragmentation.md",
      "title": "Memory Fragmentation",
      "description": "Available memory becomes divided into small, non-contiguous blocks, preventing allocation of larger objects despite sufficient total free memory.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "memory-leaks.md",
      "title": "Memory Leaks",
      "description": "Applications fail to release memory that is no longer needed, leading to gradual memory consumption and eventual performance degradation or crashes.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "memory-swapping.md",
      "title": "Memory Swapping",
      "description": "The database server runs out of physical memory and starts using disk swap space, which dramatically slows down performance.",
      "category": "Performance",
      "size": 20
    },
    {
      "id": "mental-fatigue.md",
      "title": "Mental Fatigue",
      "description": "Developers report feeling exhausted and mentally drained without accomplishing significant work, often due to cognitive overhead and inefficient workflows.",
      "category": "Culture",
      "size": 20
    },
    {
      "id": "mentor-burnout.md",
      "title": "Mentor Burnout",
      "description": "Experienced team members become overwhelmed and exhausted from excessive mentoring and knowledge transfer responsibilities.",
      "category": "Culture",
      "size": 18
    },
    {
      "id": "merge-conflicts.md",
      "title": "Merge Conflicts",
      "description": "Multiple developers frequently modify the same large functions or files, creating version control conflicts that slow development.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "micromanagement-culture.md",
      "title": "Micromanagement Culture",
      "description": "Management culture that requires approval for routine decisions, reducing team autonomy and creating unnecessary bottlenecks.",
      "category": "Culture",
      "size": 10
    },
    {
      "id": "microservice-communication-overhead.md",
      "title": "Microservice Communication Overhead",
      "description": "Excessive network communication between microservices creates latency, reduces reliability, and impacts overall system performance.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "misaligned-deliverables.md",
      "title": "Misaligned Deliverables",
      "description": "The delivered product or feature does not match the expectations or requirements of the stakeholders.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "misconfigured-connection-pools.md",
      "title": "Misconfigured Connection Pools",
      "description": "Application connection pools are improperly set up, leading to inefficient resource utilization or connection exhaustion.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "missed-deadlines.md",
      "title": "Missed Deadlines",
      "description": "Projects regularly exceed their estimated completion times and teams consistently fail to meet sprint goals and delivery commitments.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "missing-end-to-end-tests.md",
      "title": "System Integration Blindness",
      "description": "Complete user flows are not tested from start to finish, allowing workflow-breaking bugs to reach production.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "missing-rollback-strategy.md",
      "title": "Deployment Risk",
      "description": "There's no tested method to undo a deployment if things go wrong, increasing risk",
      "category": "Code",
      "size": 20
    },
    {
      "id": "misunderstanding-of-oop.md",
      "title": "Misunderstanding of OOP",
      "description": "A lack of understanding of the fundamental principles of object-oriented programming can lead to the creation of poorly designed and difficult-to-maintain code.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "mixed-coding-styles.md",
      "title": "Mixed Coding Styles",
      "description": "A situation where different parts of the codebase use different formatting, naming conventions, and design patterns.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "modernization-roi-justification-failure.md",
      "title": "Modernization ROI Justification Failure",
      "description": "Unable to build compelling business cases for legacy modernization due to hidden technical debt and unclear benefit quantification",
      "category": "Business",
      "size": 16
    },
    {
      "id": "modernization-strategy-paralysis.md",
      "title": "Modernization Strategy Paralysis",
      "description": "Teams become overwhelmed by modernization options (rewrite, refactor, replace, retire) and fail to make decisions, leaving systems in limbo",
      "category": "Management",
      "size": 20
    },
    {
      "id": "monitoring-gaps.md",
      "title": "Monitoring Gaps",
      "description": "Insufficient production monitoring and observability make it difficult to detect and diagnose issues in a timely manner, leading to longer outages and more severe consequences.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "monolithic-architecture-constraints.md",
      "title": "Monolithic Architecture Constraints",
      "description": "Large monolithic codebases become difficult to maintain, scale, and deploy as they grow in size and complexity.",
      "category": "Architecture",
      "size": 10
    },
    {
      "id": "monolithic-functions-and-classes.md",
      "title": "Monolithic Functions and Classes",
      "description": "Individual functions or classes perform too many unrelated responsibilities, making them difficult to understand and modify.",
      "category": "Architecture",
      "size": 10
    },
    {
      "id": "n-plus-one-query-problem.md",
      "title": "N+1 Query Problem",
      "description": "An application makes numerous unnecessary database calls to fetch related data where a single, more efficient query would suffice, causing significant performance degradation.",
      "category": "Data",
      "size": 16
    },
    {
      "id": "negative-brand-perception.md",
      "title": "Negative Brand Perception",
      "description": "Users associate the brand with poor quality or unreliability due to technical problems.",
      "category": "Business",
      "size": 14
    },
    {
      "id": "negative-user-feedback.md",
      "title": "Negative User Feedback",
      "description": "Users complain about slow loading times, application freezes, or other issues, indicating dissatisfaction with the application's performance or usability.",
      "category": "Business",
      "size": 10
    },
    {
      "id": "network-latency.md",
      "title": "Network Latency",
      "description": "Delays in data transmission across the network significantly increase response times and impact application performance.",
      "category": "Performance",
      "size": 14
    },
    {
      "id": "new-hire-frustration.md",
      "title": "New Hire Frustration",
      "description": "Recently hired developers experience significant frustration due to barriers preventing them from contributing effectively to the team.",
      "category": "Culture",
      "size": 14
    },
    {
      "id": "nitpicking-culture.md",
      "title": "Nitpicking Culture",
      "description": "Code reviews focus excessively on minor, insignificant details while overlooking important design and functionality issues.",
      "category": "Culture",
      "size": 12
    },
    {
      "id": "no-continuous-feedback-loop.md",
      "title": "Feedback Isolation",
      "description": "Stakeholders are not involved throughout the development process, and feedback is only gathered at the very end, leading to misaligned deliverables.",
      "category": "Communication",
      "size": 14
    },
    {
      "id": "no-formal-change-control-process.md",
      "title": "No Formal Change Control Process",
      "description": "Changes to project scope or requirements are not formally evaluated or approved, leading to uncontrolled scope creep and project delays.",
      "category": "Process",
      "size": 16
    },
    {
      "id": "null-pointer-dereferences.md",
      "title": "Null Pointer Dereferences",
      "description": "Programs attempt to access memory through null or invalid pointers, causing crashes and potential security vulnerabilities.",
      "category": "Code",
      "size": 12
    },
    {
      "id": "obsolete-technologies.md",
      "title": "Obsolete Technologies",
      "description": "The system relies on outdated tools, frameworks, or languages that make modern development practices difficult to implement.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "operational-overhead.md",
      "title": "Operational Overhead",
      "description": "A significant amount of time and resources are spent on emergency response and firefighting, rather than on planned development and innovation.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "organizational-structure-mismatch.md",
      "title": "Organizational Structure Mismatch",
      "description": "A situation where the structure of the organization does not match the architecture of the system.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "outdated-tests.md",
      "title": "Outdated Tests",
      "description": "Tests are not updated when the code changes, leading to false positives or negatives and reduced confidence.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "over-reliance-on-utility-classes.md",
      "title": "Over-Reliance on Utility Classes",
      "description": "The excessive use of utility classes with static methods can lead to a procedural style of programming and a lack of proper object-oriented design.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "overworked-teams.md",
      "title": "Overworked Teams",
      "description": "High workloads lead to burnout, mistakes, and long-term drops in quality and productivity.",
      "category": "Culture",
      "size": 20
    },
    {
      "id": "partial-bug-fixes.md",
      "title": "Partial Bug Fixes",
      "description": "Issues appear to be resolved but resurface in different contexts because the fix was not applied to all instances of the duplicated code.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "password-security-weaknesses.md",
      "title": "Password Security Weaknesses",
      "description": "Weak password policies, inadequate storage mechanisms, and poor authentication practices create security vulnerabilities.",
      "category": "Security",
      "size": 12
    },
    {
      "id": "past-negative-experiences.md",
      "title": "Past Negative Experiences",
      "description": "A situation where developers are hesitant to make changes to the codebase because of negative experiences in the past.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "perfectionist-culture.md",
      "title": "Perfectionist Culture",
      "description": "A culture of perfectionism and a reluctance to release anything that is not 100% perfect can lead to analysis paralysis and long release cycles.",
      "category": "Culture",
      "size": 20
    },
    {
      "id": "perfectionist-review-culture.md",
      "title": "Perfectionist Review Culture",
      "description": "Team culture emphasizes making code perfect through reviews rather than focusing on meaningful improvements, leading to excessive revision cycles.",
      "category": "Culture",
      "size": 11
    },
    {
      "id": "planning-credibility-issues.md",
      "title": "Planning Credibility Issues",
      "description": "Future estimates and plans are questioned or ignored due to history of inaccurate predictions and missed commitments.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "planning-dysfunction.md",
      "title": "Planning Dysfunction",
      "description": "Project planning processes fail to create realistic timelines, allocate resources effectively, or account for project complexities and risks.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "poor-caching-strategy.md",
      "title": "Poor Caching Strategy",
      "description": "Data that could be cached is fetched from the source on every request, adding unnecessary overhead and increasing latency.",
      "category": "Performance",
      "size": 10
    },
    {
      "id": "poor-communication.md",
      "title": "Communication Breakdown",
      "description": "Collaboration breaks down as developers become isolated and less willing to engage with peers.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "poor-contract-design.md",
      "title": "Poor Contract Design",
      "description": "Legal agreements and contracts don't reflect project realities, technical requirements, or allow for necessary flexibility during development.",
      "category": "Management",
      "size": 16
    },
    {
      "id": "poor-documentation.md",
      "title": "Information Decay",
      "description": "System documentation is outdated, incomplete, inaccurate, or difficult to find and use effectively.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "poor-domain-model.md",
      "title": "Poor Domain Model",
      "description": "Core business concepts are poorly understood or reflected in the system, leading to fragile logic and miscommunication",
      "category": "Architecture",
      "size": 16
    },
    {
      "id": "poor-encapsulation.md",
      "title": "Poor Encapsulation",
      "description": "Data and the behavior that acts on that data are not bundled together in a single, cohesive unit, leading to a lack of data hiding and a high degree of coupling.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "poor-interfaces-between-applications.md",
      "title": "Poor Interfaces Between Applications",
      "description": "Disconnected or poorly defined interfaces lead to fragile integrations and inconsistent data",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "poor-naming-conventions.md",
      "title": "Poor Naming Conventions",
      "description": "Variables, functions, classes, and other code elements are named in ways that don't clearly communicate their purpose or meaning.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "poor-operational-concept.md",
      "title": "Poor Operational Concept",
      "description": "Lack of planning for monitoring, maintenance, or support leads to post-launch instability",
      "category": "Operations",
      "size": 20
    },
    {
      "id": "poor-planning.md",
      "title": "Poor Planning",
      "description": "Teams do not have clear plans or realistic estimates of the work involved, leading to project delays and resource allocation problems.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "poor-project-control.md",
      "title": "Poor Project Control",
      "description": "Project progress is not monitored effectively, allowing problems to go unnoticed until recovery becomes difficult or impossible.",
      "category": "Management",
      "size": 10
    },
    {
      "id": "poor-system-environment.md",
      "title": "Poor System Environment",
      "description": "The system is deployed in an unstable, misconfigured, or unsuitable environment that causes outages, performance issues, and operational difficulties.",
      "category": "Operations",
      "size": 20
    },
    {
      "id": "poor-teamwork.md",
      "title": "Team Dysfunction",
      "description": "Team members work in isolation, resist collaboration, or lack mutual respect, reducing overall effectiveness.",
      "category": "Communication",
      "size": 12
    },
    {
      "id": "poor-test-coverage.md",
      "title": "Poor Test Coverage",
      "description": "Critical parts of the codebase are not covered by tests, creating blind spots in quality assurance.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "poor-user-experience-ux-design.md",
      "title": "Poor User Experience (UX) Design",
      "description": "The application is difficult to use, confusing, or does not meet user needs.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "poorly-defined-responsibilities.md",
      "title": "Poorly Defined Responsibilities",
      "description": "Modules or classes are not designed with a single, clear responsibility, leading to confusion and tight coupling.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "power-struggles.md",
      "title": "Power Struggles",
      "description": "Internal conflicts between departments or managers interfere with decision-making and project progress.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "premature-technology-introduction.md",
      "title": "Premature Technology Introduction",
      "description": "New frameworks, tools, or platforms are introduced without proper evaluation, adding risk and learning overhead to projects.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "priority-thrashing.md",
      "title": "Priority Thrashing",
      "description": "Work priorities change frequently and unexpectedly, causing constant task switching and disrupting planned work flow.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "procedural-background.md",
      "title": "Procedural Background",
      "description": "Developers with a background in procedural programming may struggle to adapt to an object-oriented way of thinking, leading to the creation of procedural-style code in an object-oriented language.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "procedural-programming-in-oop-languages.md",
      "title": "Procedural Programming in OOP Languages",
      "description": "Code is written in a procedural style within object-oriented languages, leading to large, monolithic functions and poor encapsulation.",
      "category": "Architecture",
      "size": 10
    },
    {
      "id": "process-design-flaws.md",
      "title": "Process Design Flaws",
      "description": "Development processes are poorly designed, creating inefficiencies, bottlenecks, and obstacles to productive work.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "procrastination-on-complex-tasks.md",
      "title": "Procrastination on Complex Tasks",
      "description": "Difficult or cognitively demanding work is consistently postponed in favor of easier, more immediately gratifying tasks.",
      "category": "Culture",
      "size": 20
    },
    {
      "id": "product-direction-chaos.md",
      "title": "Product Direction Chaos",
      "description": "Multiple stakeholders provide conflicting priorities without clear product leadership, causing team confusion and wasted effort.",
      "category": "Business",
      "size": 12
    },
    {
      "id": "project-authority-vacuum.md",
      "title": "Project Authority Vacuum",
      "description": "Critical projects lack sufficient organizational backing and executive sponsorship to overcome resistance and secure resources.",
      "category": "Management",
      "size": 10
    },
    {
      "id": "project-resource-constraints.md",
      "title": "Project Resource Constraints",
      "description": "Projects cannot obtain necessary budget, personnel, or organizational resources due to poor planning or competing priorities.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "quality-blind-spots.md",
      "title": "Quality Blind Spots",
      "description": "Critical system behaviors and failure modes remain undetected due to gaps in testing coverage and verification practices.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "quality-compromises.md",
      "title": "Quality Compromises",
      "description": "Quality standards are deliberately lowered or shortcuts are taken to meet deadlines, budgets, or other constraints, creating long-term problems.",
      "category": "Code",
      "size": 16
    },
    {
      "id": "quality-degradation.md",
      "title": "Quality Degradation",
      "description": "System quality decreases over time due to accumulating technical debt, shortcuts, and insufficient quality practices.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "queries-that-prevent-index-usage.md",
      "title": "Queries That Prevent Index Usage",
      "description": "The way a query is written can prevent the database from using an available index, forcing slower full-table scans or less efficient index scans.",
      "category": "Performance",
      "size": 10
    },
    {
      "id": "race-conditions.md",
      "title": "Race Conditions",
      "description": "Multiple threads access shared resources simultaneously without proper synchronization, leading to unpredictable behavior and data corruption.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "rapid-prototyping-becoming-production.md",
      "title": "Rapid Prototyping Becoming Production",
      "description": "Code written quickly for prototypes or proof-of-concepts ends up in production systems without proper engineering practices.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "rapid-system-changes.md",
      "title": "Rapid System Changes",
      "description": "Frequent modifications to system architecture, APIs, or core functionality outpace documentation and team understanding.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "rapid-team-growth.md",
      "title": "Rapid Team Growth",
      "description": "Teams expand in size quickly without adequate preparation, overwhelming existing infrastructure and support systems.",
      "category": "Management",
      "size": 12
    },
    {
      "id": "rate-limiting-issues.md",
      "title": "Rate Limiting Issues",
      "description": "Rate limiting mechanisms are misconfigured, too restrictive, or ineffective, causing legitimate requests to be blocked or failing to prevent abuse.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "reduced-code-submission-frequency.md",
      "title": "Reduced Code Submission Frequency",
      "description": "Developers batch multiple changes together or delay submissions to avoid frequent code review cycles, reducing feedback quality and integration frequency.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "reduced-feature-quality.md",
      "title": "Reduced Feature Quality",
      "description": "Less time is available for polish and refinement of delivered features, resulting in lower-quality user experiences and functionality.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "reduced-individual-productivity.md",
      "title": "Reduced Individual Productivity",
      "description": "Individual developers complete fewer tasks and take longer to resolve problems despite maintaining the same work effort.",
      "category": "Code",
      "size": 12
    },
    {
      "id": "reduced-innovation.md",
      "title": "Reduced Innovation",
      "description": "Teams become resistant to new ideas and focus only on the bare minimum to get by.",
      "category": "Business",
      "size": 10
    },
    {
      "id": "reduced-predictability.md",
      "title": "Reduced Predictability",
      "description": "Development timelines, outcomes, and system behavior become difficult to predict accurately, making planning and expectations management challenging.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "reduced-review-participation.md",
      "title": "Reduced Review Participation",
      "description": "Many team members avoid participating in code reviews, concentrating review burden on a few individuals and reducing coverage.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "reduced-team-flexibility.md",
      "title": "Reduced Team Flexibility",
      "description": "The team's ability to adapt to changing requirements, reassign work, or respond to unexpected challenges is significantly limited.",
      "category": "Dependencies",
      "size": 20
    },
    {
      "id": "reduced-team-productivity.md",
      "title": "Reduced Team Productivity",
      "description": "The overall output and effectiveness of the development team decreases due to various systemic issues and inefficiencies.",
      "category": "Performance",
      "size": 10
    },
    {
      "id": "refactoring-avoidance.md",
      "title": "Refactoring Avoidance",
      "description": "The development team actively avoids refactoring the codebase, even when they acknowledge it's necessary, due to fear of introducing new bugs.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "regression-bugs.md",
      "title": "Regression Bugs",
      "description": "New features or fixes inadvertently break existing functionality that was previously working correctly.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "regulatory-compliance-drift.md",
      "title": "Regulatory Compliance Drift",
      "description": "Legacy systems fall behind evolving regulatory requirements, creating compliance gaps that are expensive and risky to address",
      "category": "Management",
      "size": 10
    },
    {
      "id": "release-anxiety.md",
      "title": "Release Anxiety",
      "description": "The development team is anxious and stressed about deployments due to the high risk of failure and the pressure to get it right.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "release-instability.md",
      "title": "Release Instability",
      "description": "Production releases are frequently unstable, causing disruptions for users and requiring immediate attention from the development team.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "requirements-ambiguity.md",
      "title": "Requirements Ambiguity",
      "description": "System requirements are unclear, incomplete, or open to multiple interpretations, leading to misaligned implementations and rework.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "resistance-to-change.md",
      "title": "Resistance to Change",
      "description": "Teams are hesitant to refactor or improve parts of the system due to perceived risk and effort, leading to stagnation.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "resource-allocation-failures.md",
      "title": "Resource Allocation Failures",
      "description": "Objects, connections, file handles, or other system resources are allocated but never properly deallocated or closed, leading to resource exhaustion.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "resource-contention.md",
      "title": "Resource Contention",
      "description": "The server is overloaded, and the application is competing for limited resources like CPU, memory, or I/O.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "resource-waste.md",
      "title": "Resource Waste",
      "description": "Available resources are not utilized effectively, leading to underutilization while other areas remain constrained or overloaded.",
      "category": "Performance",
      "size": 20
    },
    {
      "id": "rest-api-design-issues.md",
      "title": "REST API Design Issues",
      "description": "Poor REST API design violates REST principles, creates usability problems, and leads to inefficient client-server interactions.",
      "category": "Architecture",
      "size": 16
    },
    {
      "id": "review-bottlenecks.md",
      "title": "Review Bottlenecks",
      "description": "The code review process becomes a significant bottleneck, delaying the delivery of new features and bug fixes.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "review-process-avoidance.md",
      "title": "Review Process Avoidance",
      "description": "Team members actively seek ways to bypass or minimize code review requirements, undermining the quality assurance process.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "review-process-breakdown.md",
      "title": "Review Process Breakdown",
      "description": "Code review practices fail to identify critical issues, provide meaningful feedback, or improve code quality due to systemic process failures.",
      "category": "Code",
      "size": 14
    },
    {
      "id": "reviewer-anxiety.md",
      "title": "Reviewer Anxiety",
      "description": "Team members feel uncertain and anxious about conducting code reviews, leading to avoidance or superficial review practices.",
      "category": "Culture",
      "size": 20
    },
    {
      "id": "reviewer-inexperience.md",
      "title": "Reviewer Inexperience",
      "description": "Reviewers lack the experience to identify deeper issues, so they focus on what they understand.",
      "category": "Code",
      "size": 14
    },
    {
      "id": "ripple-effect-of-changes.md",
      "title": "Ripple Effect of Changes",
      "description": "A small change in one part of the system requires modifications in many other seemingly unrelated parts, indicating high coupling.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "rushed-approvals.md",
      "title": "Rushed Approvals",
      "description": "Pull requests are approved quickly without thorough examination due to time pressure or process issues.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "scaling-inefficiencies.md",
      "title": "Scaling Inefficiencies",
      "description": "A situation where it is difficult or impossible to scale different parts of a system independently.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "schema-evolution-paralysis.md",
      "title": "Schema Evolution Paralysis",
      "description": "Database schema cannot be modified to support new requirements due to extensive dependencies and lack of migration tooling",
      "category": "Code",
      "size": 20
    },
    {
      "id": "scope-change-resistance.md",
      "title": "Scope Change Resistance",
      "description": "Necessary changes to project scope are avoided or resisted due to process constraints, contract limitations, or organizational inertia.",
      "category": "Management",
      "size": 17
    },
    {
      "id": "scope-creep.md",
      "title": "Scope Creep",
      "description": "Project requirements expand continuously without proper control or impact analysis, threatening timelines, budgets, and the original objectives.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "second-system-effect.md",
      "title": "Second-System Effect",
      "description": "Lessons from an old system lead to overcompensation, creating bloated or overly ambitious designs",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "secret-management-problems.md",
      "title": "Secret Management Problems",
      "description": "Inadequate handling of sensitive credentials and secrets creates security vulnerabilities and operational challenges.",
      "category": "Operations",
      "size": 18
    },
    {
      "id": "serialization-deserialization-bottlenecks.md",
      "title": "Serialization/Deserialization Bottlenecks",
      "description": "Inefficient serialization and deserialization of data creates performance bottlenecks in API communications and data persistence operations.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "service-discovery-failures.md",
      "title": "Service Discovery Failures",
      "description": "Service discovery mechanisms fail to locate or connect to services, causing communication failures and system instability in distributed architectures.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "service-timeouts.md",
      "title": "Service Timeouts",
      "description": "Services fail to complete requests within an acceptable time limit, causing errors, cascading failures, and system instability.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "session-management-issues.md",
      "title": "Session Management Issues",
      "description": "Poor session handling creates security vulnerabilities through session hijacking, fixation, or improper lifecycle management.",
      "category": "Security",
      "size": 14
    },
    {
      "id": "shadow-systems.md",
      "title": "Shadow Systems",
      "description": "Alternative solutions developed outside official channels undermine standardization and create hidden dependencies.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "shared-database.md",
      "title": "Shared Database",
      "description": "A situation where multiple services or components share a single database.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "shared-dependencies.md",
      "title": "Shared Dependencies",
      "description": "A situation where multiple components or services share a common set of libraries and frameworks.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "short-term-focus.md",
      "title": "Short-Term Focus",
      "description": "Management prioritizes immediate feature delivery over long-term code health and architectural improvements, creating sustainability issues.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "silent-data-corruption.md",
      "title": "Silent Data Corruption",
      "description": "Data becomes corrupted without triggering errors or alerts, leading to incorrect results and loss of data integrity.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "single-entry-point-design.md",
      "title": "Single Entry Point Design",
      "description": "A design where all requests to a system must go through a single object or component.",
      "category": "Architecture",
      "size": 12
    },
    {
      "id": "single-points-of-failure.md",
      "title": "Single Points of Failure",
      "description": "Progress is blocked when specific knowledge holders or system components are unavailable, creating critical dependencies.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "skill-development-gaps.md",
      "title": "Skill Development Gaps",
      "description": "Team members don't develop expertise in important technologies or domains due to avoidance, specialization, or inadequate learning opportunities.",
      "category": "Team",
      "size": 20
    },
    {
      "id": "slow-application-performance.md",
      "title": "Slow Application Performance",
      "description": "User-facing features that rely on the API feel sluggish or unresponsive.",
      "category": "Performance",
      "size": 20
    },
    {
      "id": "slow-database-queries.md",
      "title": "Slow Database Queries",
      "description": "Application performance degrades due to inefficient data retrieval from the database.",
      "category": "Performance",
      "size": 14
    },
    {
      "id": "slow-development-velocity.md",
      "title": "Slow Development Velocity",
      "description": "The team consistently fails to deliver features and bug fixes at a predictable and acceptable pace, with overall productivity systematically declining.",
      "category": "Business",
      "size": 15
    },
    {
      "id": "slow-feature-development.md",
      "title": "Slow Feature Development",
      "description": "The pace of developing and delivering new features is consistently slow, often due to the complexity and fragility of the existing codebase.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "slow-incident-resolution.md",
      "title": "Slow Incident Resolution",
      "description": "Problems and outages take excessive time to diagnose and resolve, prolonging business impact and user frustration.",
      "category": "Operations",
      "size": 20
    },
    {
      "id": "slow-knowledge-transfer.md",
      "title": "Slow Knowledge Transfer",
      "description": "A situation where it takes a long time for new team members to become productive.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "slow-response-times-for-lists.md",
      "title": "Slow Response Times for Lists",
      "description": "Web pages or API endpoints that display lists of items are significantly slower to load than those that display single items, often due to inefficient data fetching.",
      "category": "Data",
      "size": 20
    },
    {
      "id": "spaghetti-code.md",
      "title": "Spaghetti Code",
      "description": "Code with tangled, unstructured logic that is nearly impossible to understand, debug, or modify safely.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "sql-injection-vulnerabilities.md",
      "title": "SQL Injection Vulnerabilities",
      "description": "Inadequate input sanitization allows attackers to inject malicious SQL code, potentially compromising database security and data integrity.",
      "category": "Data",
      "size": 10
    },
    {
      "id": "stack-overflow-errors.md",
      "title": "Stack Overflow Errors",
      "description": "Programs exceed the allocated stack space due to excessive recursion or large local variables, causing application crashes.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "staff-availability-issues.md",
      "title": "Staff Availability Issues",
      "description": "Needed roles remain unfilled or employees are overbooked, reducing execution capacity and project progress.",
      "category": "Performance",
      "size": 20
    },
    {
      "id": "stagnant-architecture.md",
      "title": "Stagnant Architecture",
      "description": "The system architecture doesn't evolve to meet changing business needs because major refactoring is consistently avoided.",
      "category": "Architecture",
      "size": 16
    },
    {
      "id": "stakeholder-confidence-loss.md",
      "title": "Stakeholder Confidence Loss",
      "description": "Business partners lose trust in the development team's ability to deliver on commitments, creating tension and reduced collaboration.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "stakeholder-developer-communication-gap.md",
      "title": "Stakeholder-Developer Communication Gap",
      "description": "A persistent misunderstanding between what stakeholders expect and what the development team builds, leading to rework and dissatisfaction.",
      "category": "Communication",
      "size": 10
    },
    {
      "id": "stakeholder-dissatisfaction.md",
      "title": "Stakeholder Dissatisfaction",
      "description": "Business stakeholders become unhappy with project outcomes, development speed, or communication quality.",
      "category": "Communication",
      "size": 12
    },
    {
      "id": "stakeholder-frustration.md",
      "title": "Stakeholder Frustration",
      "description": "Business stakeholders become frustrated with development progress, quality, or communication, leading to strained relationships and reduced support.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "strangler-fig-pattern-failures.md",
      "title": "Strangler Fig Pattern Failures",
      "description": "Incremental modernization using the strangler fig pattern stalls due to complex interdependencies and data consistency challenges",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "style-arguments-in-code-reviews.md",
      "title": "Style Arguments in Code Reviews",
      "description": "A situation where a significant amount of time in code reviews is spent debating trivial style issues instead of focusing on logic and design.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "suboptimal-solutions.md",
      "title": "Suboptimal Solutions",
      "description": "Delivered solutions work but are inefficient, difficult to use, or don't fully address the underlying problems they were meant to solve.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "superficial-code-reviews.md",
      "title": "Superficial Code Reviews",
      "description": "Code reviews focus only on surface-level issues like formatting and style while missing important design, logic, or security problems.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "synchronization-problems.md",
      "title": "Synchronization Problems",
      "description": "Updates to one copy of duplicated logic don't get applied to other copies, causing divergent behavior across the system.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "system-integration-blindness.md",
      "title": "System Integration Blindness",
      "description": "Components work correctly in isolation but fail when integrated, revealing gaps in end-to-end system understanding.",
      "category": "Architecture",
      "size": 10
    },
    {
      "id": "system-outages.md",
      "title": "System Outages",
      "description": "Service interruptions and system failures occur frequently, causing business disruption and user frustration.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "system-stagnation.md",
      "title": "System Stagnation",
      "description": "Software systems remain unchanged and fail to evolve to meet new requirements, technologies, or business needs over extended periods.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "tacit-knowledge.md",
      "title": "Tacit Knowledge",
      "description": "Knowledge that is difficult to transfer to another person by means of writing it down or verbalizing it.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "tangled-cross-cutting-concerns.md",
      "title": "Tangled Cross-Cutting Concerns",
      "description": "A situation where cross-cutting concerns, such as logging, security, and transactions, are tightly coupled with the business logic.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "task-queues-backing-up.md",
      "title": "Task Queues Backing Up",
      "description": "Asynchronous jobs or messages take longer to process, causing queues to grow and delaying critical operations.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "team-churn-impact.md",
      "title": "Team Churn Impact",
      "description": "Over time, as developers join and leave the team, they bring inconsistent practices and knowledge gaps that degrade code quality.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "team-confusion.md",
      "title": "Team Confusion",
      "description": "Team members are unclear about project goals, priorities, responsibilities, or processes, leading to misaligned efforts and wasted work.",
      "category": "Communication",
      "size": 18
    },
    {
      "id": "team-coordination-issues.md",
      "title": "Team Coordination Issues",
      "description": "A situation where multiple developers or teams have difficulty working together on the same codebase.",
      "category": "Process",
      "size": 16
    },
    {
      "id": "team-demoralization.md",
      "title": "Team Demoralization",
      "description": "Team members lose motivation, confidence, and enthusiasm for their work due to persistent problems or organizational issues.",
      "category": "Culture",
      "size": 20
    },
    {
      "id": "team-dysfunction.md",
      "title": "Team Dysfunction",
      "description": "Team members cannot collaborate effectively due to conflicting goals, interpersonal issues, or structural problems that prevent coordinated effort.",
      "category": "Management",
      "size": 15
    },
    {
      "id": "team-members-not-engaged-in-review-process.md",
      "title": "Team Members Not Engaged in Review Process",
      "description": "Code reviews are often assigned to the same people, or reviewers do not provide meaningful feedback, leading to a bottleneck and reduced quality.",
      "category": "Communication",
      "size": 10
    },
    {
      "id": "team-silos.md",
      "title": "Team Silos",
      "description": "Development teams or individual developers work in isolation, leading to duplicated effort, inconsistent solutions, and a lack of knowledge sharing.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "technical-architecture-limitations.md",
      "title": "Technical Architecture Limitations",
      "description": "System architecture design creates constraints that limit performance, scalability, maintainability, or development velocity.",
      "category": "Architecture",
      "size": 12
    },
    {
      "id": "technology-isolation.md",
      "title": "Technology Isolation",
      "description": "The system becomes increasingly isolated from modern technology stacks, limiting ability to attract talent and leverage new capabilities.",
      "category": "Code",
      "size": 12
    },
    {
      "id": "technology-lock-in.md",
      "title": "Technology Lock-In",
      "description": "A situation where it is difficult or impossible to switch to a new technology because of the high cost or effort involved.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "technology-stack-fragmentation.md",
      "title": "Technology Stack Fragmentation",
      "description": "Legacy systems create isolated technology islands that prevent standardization and increase operational complexity across the organization",
      "category": "Code",
      "size": 20
    },
    {
      "id": "test-debt.md",
      "title": "Test Debt",
      "description": "The accumulated risk from inadequate or neglected quality assurance, leading to a fragile product and slow development velocity.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "testing-complexity.md",
      "title": "Testing Complexity",
      "description": "Quality assurance must verify the same functionality in multiple locations, which increases the testing effort and the risk of missing bugs.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "testing-environment-fragility.md",
      "title": "Testing Environment Fragility",
      "description": "Testing infrastructure is unreliable, difficult to maintain, and fails to accurately represent production conditions, undermining test effectiveness.",
      "category": "Operations",
      "size": 20
    },
    {
      "id": "thread-pool-exhaustion.md",
      "title": "Thread Pool Exhaustion",
      "description": "All available threads in the thread pool are consumed by long-running or blocked operations, preventing new tasks from being processed.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "tight-coupling-issues.md",
      "title": "Tight Coupling Issues",
      "description": "Components are overly dependent on each other, making changes difficult and reducing system flexibility and maintainability.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "time-pressure.md",
      "title": "Time Pressure",
      "description": "Teams are forced to take shortcuts to meet immediate deadlines, deferring proper solutions and rushing important tasks like code reviews.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "tool-limitations.md",
      "title": "Tool Limitations",
      "description": "Inadequate development tools slow down common tasks and reduce developer productivity and satisfaction.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "unbounded-data-growth.md",
      "title": "Unbounded Data Growth",
      "description": "Data structures, caches, or databases grow indefinitely without proper pruning, size limits, or archiving strategies.",
      "category": "Code",
      "size": 14
    },
    {
      "id": "unbounded-data-structures.md",
      "title": "Unbounded Data Structures",
      "description": "Data structures that grow indefinitely without proper pruning or size limits, leading to memory exhaustion and performance degradation.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "unclear-documentation-ownership.md",
      "title": "Unclear Documentation Ownership",
      "description": "No clear responsibility for maintaining documentation leads to outdated, inconsistent, or missing information.",
      "category": "Communication",
      "size": 10
    },
    {
      "id": "unclear-goals-and-priorities.md",
      "title": "Unclear Goals and Priorities",
      "description": "Constantly shifting priorities and lack of clear direction lead to a sense of futility among team members.",
      "category": "Communication",
      "size": 14
    },
    {
      "id": "unclear-sharing-expectations.md",
      "title": "Unclear Sharing Expectations",
      "description": "A situation where it is not clear what information should be shared with the rest of the team.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "uncontrolled-codebase-growth.md",
      "title": "Uncontrolled Codebase Growth",
      "description": "A situation where a codebase grows in size and complexity without any control or planning.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "undefined-code-style-guidelines.md",
      "title": "Undefined Code Style Guidelines",
      "description": "The team lacks clear, agreed-upon coding standards, resulting in subjective stylistic feedback and inconsistent code.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "uneven-work-flow.md",
      "title": "Uneven Work Flow",
      "description": "Work progresses in irregular fits and starts rather than flowing smoothly through the development process.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "uneven-workload-distribution.md",
      "title": "Uneven Workload Distribution",
      "description": "Work is not distributed fairly or effectively across team members, leading to some being overloaded while others are underutilized.",
      "category": "Performance",
      "size": 20
    },
    {
      "id": "unmotivated-employees.md",
      "title": "Unmotivated Employees",
      "description": "Team members lack engagement and enthusiasm, contributing minimally to project success and affecting overall team morale.",
      "category": "Culture",
      "size": 12
    },
    {
      "id": "unoptimized-file-access.md",
      "title": "Unoptimized File Access",
      "description": "Applications read or write files inefficiently, leading to excessive disk I/O and slow performance.",
      "category": "Performance",
      "size": 16
    },
    {
      "id": "unpredictable-system-behavior.md",
      "title": "Unpredictable System Behavior",
      "description": "Changes in one part of the system have unexpected side effects in seemingly unrelated areas due to hidden dependencies.",
      "category": "Architecture",
      "size": 14
    },
    {
      "id": "unproductive-meetings.md",
      "title": "Unproductive Meetings",
      "description": "Meetings that consume significant time but yield little progress or concrete decisions.",
      "category": "Communication",
      "size": 16
    },
    {
      "id": "unrealistic-deadlines.md",
      "title": "Unrealistic Deadlines",
      "description": "Management sets aggressive deadlines that do not account for the actual effort required, leading to compromised quality and unsustainable work practices.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "unrealistic-schedule.md",
      "title": "Unrealistic Schedule",
      "description": "Project timelines are based on optimistic assumptions rather than realistic estimates, leading to stress and shortcuts.",
      "category": "Management",
      "size": 12
    },
    {
      "id": "unreleased-resources.md",
      "title": "Unreleased Resources",
      "description": "Objects, connections, file handles, or other system resources are allocated but never properly deallocated or closed.",
      "category": "Code",
      "size": 12
    },
    {
      "id": "unused-indexes.md",
      "title": "Unused Indexes",
      "description": "The database has indexes that are never used by any queries, which still consume storage space and add overhead to write operations.",
      "category": "Code",
      "size": 12
    },
    {
      "id": "upstream-timeouts.md",
      "title": "Upstream Timeouts",
      "description": "Services that consume an API fail because they do not receive a response within their configured timeout window.",
      "category": "Code",
      "size": 10
    },
    {
      "id": "user-confusion.md",
      "title": "User Confusion",
      "description": "End users encounter different behavior for what should be identical operations, leading to frustration and a loss of trust in the system.",
      "category": "Requirements",
      "size": 20
    },
    {
      "id": "user-frustration.md",
      "title": "User Frustration",
      "description": "Users become dissatisfied with system reliability, usability, or performance, leading to decreased adoption and negative feedback.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "user-trust-erosion.md",
      "title": "User Trust Erosion",
      "description": "Frequent issues and emergency fixes damage user confidence in the system's reliability, leading to a decline in user engagement and satisfaction.",
      "category": "Business",
      "size": 18
    },
    {
      "id": "vendor-dependency-entrapment.md",
      "title": "Vendor Dependency Entrapment",
      "description": "Legacy systems become trapped by discontinued vendor products, forcing expensive custom support contracts or complete system replacement",
      "category": "Code",
      "size": 12
    },
    {
      "id": "vendor-dependency.md",
      "title": "Vendor Dependency",
      "description": "Excessive reliance on external vendors or suppliers creates risks when they become unavailable, change terms, or fail to meet expectations.",
      "category": "Dependencies",
      "size": 20
    },
    {
      "id": "vendor-lock-in.md",
      "title": "Vendor Lock-In",
      "description": "System is overly dependent on a specific vendor's tools or APIs, limiting future options",
      "category": "Code",
      "size": 12
    },
    {
      "id": "vendor-relationship-strain.md",
      "title": "Vendor Relationship Strain",
      "description": "Tensions and conflicts arise between the organization and external vendors due to misaligned expectations, poor communication, or contract issues.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "virtual-memory-thrashing.md",
      "title": "Virtual Memory Thrashing",
      "description": "System constantly swaps pages between physical memory and disk, causing severe performance degradation due to excessive paging activity.",
      "category": "Code",
      "size": 14
    },
    {
      "id": "wasted-development-effort.md",
      "title": "Wasted Development Effort",
      "description": "Significant development work is abandoned, reworked, or becomes obsolete due to poor planning, changing requirements, or inefficient processes.",
      "category": "Performance",
      "size": 20
    },
    {
      "id": "work-blocking.md",
      "title": "Work Blocking",
      "description": "Development tasks cannot proceed without pending approvals, creating bottlenecks and delays in the development process.",
      "category": "Management",
      "size": 18
    },
    {
      "id": "work-queue-buildup.md",
      "title": "Work Queue Buildup",
      "description": "Tasks accumulate in queues waiting for bottleneck resources or processes, creating delays and reducing overall system throughput.",
      "category": "Performance",
      "size": 20
    },
    {
      "id": "workaround-culture.md",
      "title": "Workaround Culture",
      "description": "Teams implement increasingly complex workarounds rather than fixing root issues, creating layers of technical debt.",
      "category": "Code",
      "size": 20
    }
  ],
  "links": [
    {
      "source": "abi-compatibility-issues.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "abi-compatibility-issues.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "abi-compatibility-issues.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "abi-compatibility-issues.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "bloated-class.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "changing-project-scope.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "development-disruption.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "feature-creep.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "feature-gaps.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "gold-plating.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "inconsistent-knowledge-acquisition.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "information-decay.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "language-barriers.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "lock-contention.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "merge-conflicts.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "planning-credibility-issues.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "power-struggles.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "release-instability.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "schema-evolution-paralysis.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "scope-change-resistance.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "system-outages.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "team-silos.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "technology-stack-fragmentation.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "testing-environment-fragility.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "user-confusion.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "user-frustration.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "architectural-mismatch.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "architectural-mismatch.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "feature-gaps.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "user-confusion.md"
    },
    {
      "source": "author-frustration.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "author-frustration.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "author-frustration.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "author-frustration.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "author-frustration.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "author-frustration.md",
      "target": "clever-code.md"
    },
    {
      "source": "author-frustration.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "author-frustration.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "author-frustration.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "author-frustration.md",
      "target": "extended-review-cycles.md"
    },
    {
      "source": "author-frustration.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "author-frustration.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "author-frustration.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "author-frustration.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "author-frustration.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "author-frustration.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "author-frustration.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "author-frustration.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "author-frustration.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "author-frustration.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "author-frustration.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "author-frustration.md",
      "target": "procedural-background.md"
    },
    {
      "source": "author-frustration.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "author-frustration.md",
      "target": "release-instability.md"
    },
    {
      "source": "author-frustration.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "author-frustration.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "author-frustration.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "author-frustration.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "authorization-flaws.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "blame-culture.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "blame-culture.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "blame-culture.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "cache-invalidation-problems.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "cache-invalidation-problems.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "cache-invalidation-problems.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "cache-invalidation-problems.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "cache-invalidation-problems.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "cache-invalidation-problems.md",
      "target": "race-conditions.md"
    },
    {
      "source": "cache-invalidation-problems.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "complex-implementation-paths.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "development-disruption.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "gold-plating.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "legacy-configuration-management-chaos.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "microservice-communication-overhead.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "poor-communication.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "poor-planning.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "release-instability.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "schema-evolution-paralysis.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "skill-development-gaps.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "user-confusion.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "shared-database.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "user-confusion.md"
    },
    {
      "source": "change-management-chaos.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "change-management-chaos.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "changing-project-scope.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "changing-project-scope.md",
      "target": "gold-plating.md"
    },
    {
      "source": "changing-project-scope.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "changing-project-scope.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "changing-project-scope.md",
      "target": "maintenance-cost-increase.md"
    },
    {
      "source": "code-duplication.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "code-duplication.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "code-duplication.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "code-duplication.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "code-duplication.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "code-duplication.md",
      "target": "procedural-background.md"
    },
    {
      "source": "code-duplication.md",
      "target": "shared-database.md"
    },
    {
      "source": "code-duplication.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "code-duplication.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "code-duplication.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "code-duplication.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "communication-breakdown.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "communication-risk-within-project.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "communication-risk-within-project.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "communication-risk-within-project.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "competing-priorities.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "competing-priorities.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "competing-priorities.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "competing-priorities.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "competing-priorities.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "bloated-class.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "circular-dependency-problems.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "clever-code.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "context-switching-overhead.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "development-disruption.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "graphql-complexity-issues.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "high-coupling-low-cohesion.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "high-turnover.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "inconsistent-codebase.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "inconsistent-knowledge-acquisition.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "increased-risk-of-bugs.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "inexperienced-developers.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "information-decay.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "information-fragmentation.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "modernization-roi-justification-failure.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "poor-domain-model.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "procrastination-on-complex-tasks.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "ripple-effect-of-changes.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "schema-evolution-paralysis.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "scope-creep.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "second-system-effect.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "shared-database.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "strangler-fig-pattern-failures.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "technology-stack-fragmentation.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "user-confusion.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "vendor-dependency.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "procrastination-on-complex-tasks.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "configuration-chaos.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "procedural-background.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "copy-paste-programming.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "copy-paste-programming.md",
      "target": "procedural-background.md"
    },
    {
      "source": "copy-paste-programming.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "copy-paste-programming.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "high-turnover.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "schema-evolution-paralysis.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "shared-database.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "system-outages.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "user-confusion.md"
    },
    {
      "source": "cv-driven-development.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "cv-driven-development.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "cv-driven-development.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "cv-driven-development.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "complex-implementation-paths.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "dma-coherency-issues.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "high-turnover.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "implementation-starts-without-design.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "index-fragmentation.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "information-decay.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "shared-database.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "stagnant-architecture.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "data-structure-cache-inefficiency.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "data-structure-cache-inefficiency.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "database-schema-design-problems.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "database-schema-design-problems.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "development-disruption.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "duplicated-research-effort.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "duplicated-work.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "gold-plating.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "high-turnover.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "planning-credibility-issues.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "poor-communication.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "procrastination-on-complex-tasks.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "release-instability.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "reviewer-anxiety.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "team-confusion.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "team-silos.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "work-blocking.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "decision-avoidance.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "decision-avoidance.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "decision-avoidance.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "decision-avoidance.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "defensive-coding-practices.md",
      "target": "procedural-background.md"
    },
    {
      "source": "defensive-coding-practices.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "buffer-overflow-vulnerabilities.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "feature-creep.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "gold-plating.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "information-decay.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "log-spam.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "release-instability.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "team-silos.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "user-confusion.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "delayed-issue-resolution.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "delayed-issue-resolution.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "delayed-issue-resolution.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "delayed-issue-resolution.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "excessive-logging.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "long-running-database-transactions.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "team-silos.md"
    },
    {
      "source": "dependency-on-supplier.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "dependency-version-conflicts.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "dependency-version-conflicts.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "context-switching-overhead.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "decision-avoidance.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "development-disruption.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "difficult-developer-onboarding.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "duplicated-research-effort.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "duplicated-work.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "fear-of-failure.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "feature-creep.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "high-client-side-resource-consumption.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "high-resource-utilization-on-client.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "high-turnover.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "inappropriate-skillset.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "inconsistent-codebase.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "inexperienced-developers.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "large-feature-scope.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "mental-fatigue.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "mentor-burnout.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "poor-communication.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "poor-contract-design.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "power-struggles.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "priority-thrashing.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "procrastination-on-complex-tasks.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "reduced-code-submission-frequency.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "reduced-review-participation.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "reduced-team-flexibility.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "release-instability.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "resource-contention.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "reviewer-anxiety.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "stakeholder-frustration.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "team-churn-impact.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "team-confusion.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "team-demoralization.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "team-silos.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "technology-isolation.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "time-pressure.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "unproductive-meetings.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "user-confusion.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "user-frustration.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "user-trust-erosion.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "work-blocking.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "difficult-code-reuse.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "difficult-code-reuse.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "difficult-code-reuse.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "difficult-code-reuse.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "difficult-code-reuse.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "difficult-code-reuse.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "procedural-background.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "hardcoded-values.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "procedural-background.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "shared-database.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "user-confusion.md"
    },
    {
      "source": "difficulty-quantifying-benefits.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "difficulty-quantifying-benefits.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "difficulty-quantifying-benefits.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "difficulty-quantifying-benefits.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "difficulty-quantifying-benefits.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "difficulty-quantifying-benefits.md",
      "target": "procedural-background.md"
    },
    {
      "source": "difficulty-quantifying-benefits.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "duplicated-research-effort.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "duplicated-research-effort.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "duplicated-research-effort.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "duplicated-research-effort.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "duplicated-research-effort.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "duplicated-research-effort.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "duplicated-research-effort.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "duplicated-research-effort.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "duplicated-research-effort.md",
      "target": "shared-database.md"
    },
    {
      "source": "duplicated-research-effort.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "duplicated-research-effort.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "duplicated-research-effort.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "duplicated-work.md",
      "target": "shared-database.md"
    },
    {
      "source": "duplicated-work.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "environment-variable-issues.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "environment-variable-issues.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "excessive-class-size.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "excessive-class-size.md",
      "target": "procedural-background.md"
    },
    {
      "source": "excessive-object-allocation.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "excessive-object-allocation.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "extended-review-cycles.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "extended-review-cycles.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "extended-review-cycles.md",
      "target": "inadequate-initial-reviews.md"
    },
    {
      "source": "extended-review-cycles.md",
      "target": "large-pull-requests.md"
    },
    {
      "source": "extended-review-cycles.md",
      "target": "perfectionist-review-culture.md"
    },
    {
      "source": "false-sharing.md",
      "target": "shared-database.md"
    },
    {
      "source": "false-sharing.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "fear-of-conflict.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "fear-of-conflict.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "release-instability.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "user-confusion.md"
    },
    {
      "source": "feature-bloat.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "feature-creep-without-refactoring.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "feature-creep-without-refactoring.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "feature-creep.md",
      "target": "bloated-class.md"
    },
    {
      "source": "feature-creep.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "feature-creep.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "feature-creep.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "feature-creep.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "feature-creep.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "feature-creep.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "feature-creep.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "feature-creep.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "feature-creep.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "feature-creep.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "feature-creep.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "feature-creep.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "feature-creep.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "feature-creep.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "feature-creep.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "feature-creep.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "feature-creep.md",
      "target": "procedural-background.md"
    },
    {
      "source": "feature-creep.md",
      "target": "release-instability.md"
    },
    {
      "source": "feature-creep.md",
      "target": "shared-database.md"
    },
    {
      "source": "feature-creep.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "feature-creep.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "feature-creep.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "feature-creep.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "feature-creep.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "feature-creep.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "feature-creep.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "feature-factory.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "feature-factory.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "feature-factory.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "feature-factory.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "feature-gaps.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "procedural-background.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "author-frustration.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "authorization-flaws.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "cache-invalidation-problems.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "circular-dependency-problems.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "clever-code.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "dependency-version-conflicts.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "development-disruption.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "duplicated-research-effort.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "global-state-and-side-effects.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "high-turnover.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "inadequate-error-handling.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "inadequate-initial-reviews.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "inconsistent-coding-standards.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "inconsistent-knowledge-acquisition.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "increased-risk-of-bugs.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "information-decay.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "information-fragmentation.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "quality-compromises.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "race-conditions.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "rate-limiting-issues.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "reduced-code-submission-frequency.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "reduced-team-flexibility.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "regression-bugs.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "release-instability.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "strangler-fig-pattern-failures.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "system-outages.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "team-confusion.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "technology-stack-fragmentation.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "testing-environment-fragility.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "user-confusion.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "user-trust-erosion.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "cascade-delays.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "changing-project-scope.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "code-duplication.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "communication-risk-outside-project.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "competing-priorities.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "complex-implementation-paths.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "context-switching-overhead.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "decision-avoidance.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "development-disruption.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "difficult-developer-onboarding.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "duplicated-research-effort.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "duplicated-work.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "fear-of-failure.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "feature-bloat.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "feature-creep.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "feedback-isolation.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "gold-plating.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "high-turnover.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "implementation-rework.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "inadequate-code-reviews.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "inadequate-onboarding.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "inappropriate-skillset.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "inconsistent-codebase.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "inconsistent-coding-standards.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "inconsistent-knowledge-acquisition.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "increased-error-rates.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "increased-risk-of-bugs.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "inefficient-frontend-code.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "inexperienced-developers.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "information-decay.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "information-fragmentation.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "insufficient-code-review.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "insufficient-worker-capacity.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "integration-difficulties.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "large-feature-scope.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "long-release-cycles.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "long-running-database-transactions.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "lower-code-quality.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "maintenance-cost-increase.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "market-pressure.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "mentor-burnout.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "merge-conflicts.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "modernization-roi-justification-failure.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "new-hire-frustration.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "no-continuous-feedback-loop.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "no-formal-change-control-process.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "obsolete-technologies.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "planning-credibility-issues.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "poor-communication.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "poor-contract-design.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "poor-planning.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "power-struggles.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "priority-thrashing.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "procrastination-on-complex-tasks.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "product-direction-chaos.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "quality-compromises.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "reduced-code-submission-frequency.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "reduced-team-flexibility.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "release-instability.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "resource-contention.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "review-process-avoidance.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "review-process-breakdown.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "ripple-effect-of-changes.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "schema-evolution-paralysis.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "scope-creep.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "second-system-effect.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "service-timeouts.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "stakeholder-frustration.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "strangler-fig-pattern-failures.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "system-outages.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "team-churn-impact.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "team-confusion.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "team-silos.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "testing-environment-fragility.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "time-pressure.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "unclear-goals-and-priorities.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "unpredictable-system-behavior.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "unproductive-meetings.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "unrealistic-schedule.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "user-confusion.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "user-frustration.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "work-blocking.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "increased-error-rates.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "garbage-collection-pressure.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "garbage-collection-pressure.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "garbage-collection-pressure.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "garbage-collection-pressure.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "garbage-collection-pressure.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "global-state-and-side-effects.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "global-state-and-side-effects.md",
      "target": "shared-database.md"
    },
    {
      "source": "global-state-and-side-effects.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "god-object-anti-pattern.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "gold-plating.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "gold-plating.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "gold-plating.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "gold-plating.md",
      "target": "procedural-background.md"
    },
    {
      "source": "gold-plating.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "procedural-background.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "release-instability.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "high-client-side-resource-consumption.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "high-client-side-resource-consumption.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "high-client-side-resource-consumption.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "high-client-side-resource-consumption.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "high-client-side-resource-consumption.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "high-client-side-resource-consumption.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "high-client-side-resource-consumption.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "high-client-side-resource-consumption.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "procedural-background.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "development-disruption.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "high-resource-utilization-on-client.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "index-fragmentation.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "inefficient-frontend-code.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "log-spam.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "poor-planning.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "poor-teamwork.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "procedural-background.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "release-instability.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "service-timeouts.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "team-dysfunction.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "user-confusion.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "high-resource-utilization-on-client.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "high-resource-utilization-on-client.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "high-resource-utilization-on-client.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "high-resource-utilization-on-client.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "history-of-failed-changes.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "user-confusion.md"
    },
    {
      "source": "improper-event-listener-management.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "clever-code.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "gold-plating.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "legacy-configuration-management-chaos.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "maintenance-cost-increase.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "merge-conflicts.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "procedural-background.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "release-instability.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "system-outages.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "team-silos.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "user-confusion.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "clever-code.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "excessive-object-allocation.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "gold-plating.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "procedural-background.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "team-silos.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "test-debt.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "time-pressure.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "cache-invalidation-problems.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "clever-code.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "gold-plating.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "hardcoded-values.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "implementation-starts-without-design.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "logging-configuration-issues.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "release-instability.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "secret-management-problems.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "system-outages.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "user-confusion.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "inadequate-initial-reviews.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "inadequate-initial-reviews.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "inadequate-integration-tests.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "inadequate-integration-tests.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "inadequate-integration-tests.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "inadequate-integration-tests.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "inadequate-integration-tests.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "inadequate-integration-tests.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "inadequate-integration-tests.md",
      "target": "shared-database.md"
    },
    {
      "source": "inadequate-integration-tests.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "author-frustration.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "gold-plating.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "procedural-background.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "reviewer-anxiety.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "skill-development-gaps.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "team-silos.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "technology-stack-fragmentation.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "inadequate-onboarding.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "gold-plating.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "cache-invalidation-problems.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "clever-code.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "communication-risk-outside-project.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "data-protection-risk.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "database-schema-design-problems.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "development-disruption.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "inadequate-error-handling.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "information-decay.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "load-balancing-problems.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "log-spam.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "poor-planning.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "procedural-background.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "reduced-review-participation.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "release-instability.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "reviewer-inexperience.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "session-management-issues.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "shared-database.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "test-debt.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "user-confusion.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "procedural-background.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "user-confusion.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "inconsistent-knowledge-acquisition.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "inconsistent-knowledge-acquisition.md",
      "target": "clever-code.md"
    },
    {
      "source": "inconsistent-knowledge-acquisition.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "inconsistent-knowledge-acquisition.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "inconsistent-knowledge-acquisition.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "inconsistent-knowledge-acquisition.md",
      "target": "gold-plating.md"
    },
    {
      "source": "inconsistent-knowledge-acquisition.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "inconsistent-knowledge-acquisition.md",
      "target": "procedural-background.md"
    },
    {
      "source": "inconsistent-knowledge-acquisition.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "author-frustration.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "circular-dependency-problems.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "clever-code.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "defensive-coding-practices.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "dependency-version-conflicts.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "global-state-and-side-effects.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "hardcoded-values.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "inconsistent-knowledge-acquisition.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "procedural-background.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "race-conditions.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "user-confusion.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "author-frustration.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "gold-plating.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "inconsistent-coding-standards.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "individual-recognition-culture.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "load-balancing-problems.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "poor-communication.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "reduced-review-participation.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "release-instability.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "system-outages.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "user-confusion.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "vendor-dependency.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "inconsistent-quality.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "inconsistent-quality.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "inconsistent-quality.md",
      "target": "shared-database.md"
    },
    {
      "source": "inconsistent-quality.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "inconsistent-quality.md",
      "target": "user-confusion.md"
    },
    {
      "source": "incorrect-index-type.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "incorrect-index-type.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "incorrect-index-type.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "incorrect-index-type.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "incorrect-max-connection-pool-size.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "incorrect-max-connection-pool-size.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "incorrect-max-connection-pool-size.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "incorrect-max-connection-pool-size.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "incorrect-max-connection-pool-size.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "incorrect-max-connection-pool-size.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "incorrect-max-connection-pool-size.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "incorrect-max-connection-pool-size.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "capacity-mismatch.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "cascade-delays.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "context-switching-overhead.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "database-query-performance-issues.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "development-disruption.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "difficult-developer-onboarding.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "feature-bloat.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "feedback-isolation.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "garbage-collection-pressure.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "high-client-side-resource-consumption.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "high-connection-count.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "high-resource-utilization-on-client.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "high-turnover.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "inadequate-onboarding.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "increased-error-rates.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "increased-risk-of-bugs.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "inefficient-code.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "inefficient-frontend-code.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "insufficient-worker-capacity.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "load-balancing-problems.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "log-spam.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "maintenance-cost-increase.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "market-pressure.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "microservice-communication-overhead.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "n-plus-one-query-problem.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "negative-brand-perception.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "no-continuous-feedback-loop.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "poor-communication.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "priority-thrashing.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "rapid-team-growth.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "rate-limiting-issues.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "reduced-code-submission-frequency.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "regression-bugs.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "resource-contention.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "second-system-effect.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "stakeholder-dissatisfaction.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "stakeholder-frustration.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "system-outages.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "time-pressure.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "user-confusion.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "user-frustration.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "user-trust-erosion.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "vendor-dependency.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "increased-technical-shortcuts.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "increased-technical-shortcuts.md",
      "target": "release-instability.md"
    },
    {
      "source": "increased-technical-shortcuts.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "gold-plating.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "log-spam.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "procedural-background.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "shared-database.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "index-fragmentation.md",
      "target": "user-confusion.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "bloated-class.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "information-decay.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "log-spam.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "microservice-communication-overhead.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "poor-communication.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "procedural-background.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "release-instability.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "rest-api-design-issues.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "serialization-deserialization-bottlenecks.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "user-confusion.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "shared-database.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "cascade-delays.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "clever-code.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "communication-risk-outside-project.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "database-query-performance-issues.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "decision-avoidance.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "defensive-coding-practices.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "development-disruption.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "difficult-to-test-code.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "feature-bloat.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "high-turnover.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "inadequate-initial-reviews.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "inadequate-onboarding.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "inadequate-requirements-gathering.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "inadequate-test-infrastructure.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "inconsistent-codebase.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "inconsistent-coding-standards.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "incorrect-index-type.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "information-decay.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "insufficient-worker-capacity.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "large-feature-scope.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "mental-fatigue.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "n-plus-one-query-problem.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "network-latency.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "poor-communication.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "power-struggles.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "priority-thrashing.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "procedural-background.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "reduced-team-flexibility.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "release-instability.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "resource-waste.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "review-process-avoidance.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "reviewer-anxiety.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "skill-development-gaps.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "stagnant-architecture.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "team-silos.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "testing-environment-fragility.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "unoptimized-file-access.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "user-confusion.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "inefficient-frontend-code.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "inefficient-frontend-code.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "inefficient-frontend-code.md",
      "target": "procedural-background.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "bloated-class.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "individual-recognition-culture.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "shared-database.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "user-confusion.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "authentication-bypass-vulnerabilities.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "data-protection-risk.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "log-spam.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "password-security-weaknesses.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "quality-blind-spots.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "secret-management-problems.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "user-confusion.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "clever-code.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "excessive-object-allocation.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "gold-plating.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "procedural-background.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "team-silos.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "test-debt.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "time-pressure.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "inadequate-onboarding.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "incorrect-index-type.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "index-fragmentation.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "inefficient-frontend-code.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "information-decay.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "poor-planning.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "procedural-background.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "release-instability.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "team-dysfunction.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "procedural-background.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "dma-coherency-issues.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "excessive-logging.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "high-connection-count.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "high-turnover.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "increased-error-rates.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "log-spam.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "microservice-communication-overhead.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "priority-thrashing.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "race-conditions.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "ripple-effect-of-changes.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "serialization-deserialization-bottlenecks.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "service-timeouts.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "stack-overflow-errors.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "system-outages.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "user-confusion.md"
    },
    {
      "source": "interrupt-overhead.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "clever-code.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "release-instability.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "user-confusion.md"
    },
    {
      "source": "knowledge-dependency.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "knowledge-dependency.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "knowledge-dependency.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "knowledge-dependency.md",
      "target": "shared-database.md"
    },
    {
      "source": "knowledge-dependency.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "knowledge-dependency.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "individual-recognition-culture.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "shared-database.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "user-confusion.md"
    },
    {
      "source": "language-barriers.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "language-barriers.md",
      "target": "scope-change-resistance.md"
    },
    {
      "source": "large-feature-scope.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "large-feature-scope.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "large-feature-scope.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "large-feature-scope.md",
      "target": "procedural-background.md"
    },
    {
      "source": "large-feature-scope.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "bloated-class.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "clever-code.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "feature-creep.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "gold-plating.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "database-schema-design-problems.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "global-state-and-side-effects.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "high-turnover.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "information-decay.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "obsolete-technologies.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "secret-management-problems.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "user-confusion.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "procedural-background.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "user-confusion.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "procedural-background.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "clever-code.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "code-duplication.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "communication-risk-outside-project.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "complex-implementation-paths.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "data-migration-integrity-issues.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "decision-avoidance.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "dependency-on-supplier.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "development-disruption.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "difficult-to-test-code.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "feature-creep.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "feedback-isolation.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "hardcoded-values.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "high-turnover.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "inadequate-configuration-management.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "inadequate-error-handling.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "inadequate-onboarding.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "inadequate-requirements-gathering.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "inadequate-test-infrastructure.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "inappropriate-skillset.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "inconsistent-knowledge-acquisition.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "increased-manual-work.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "inexperienced-developers.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "information-decay.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "integration-difficulties.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "invisible-nature-of-technical-debt.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "language-barriers.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "legacy-configuration-management-chaos.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "legacy-system-documentation-archaeology.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "lower-code-quality.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "maintenance-cost-increase.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "mentor-burnout.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "modernization-roi-justification-failure.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "negative-brand-perception.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "no-continuous-feedback-loop.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "obsolete-technologies.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "poor-communication.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "poor-domain-model.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "procedural-background.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "rate-limiting-issues.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "reduced-team-flexibility.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "regression-bugs.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "resource-waste.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "review-process-avoidance.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "schema-evolution-paralysis.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "second-system-effect.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "secret-management-problems.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "shared-database.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "single-entry-point-design.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "slow-development-velocity.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "strangler-fig-pattern-failures.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "system-outages.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "team-churn-impact.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "team-silos.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "technology-stack-fragmentation.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "test-debt.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "testing-environment-fragility.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "user-confusion.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "user-frustration.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "vendor-dependency-entrapment.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "vendor-dependency.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "legacy-system-documentation-archaeology.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "legacy-system-documentation-archaeology.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "legacy-system-documentation-archaeology.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "legacy-system-documentation-archaeology.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "legacy-system-documentation-archaeology.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "legacy-system-documentation-archaeology.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "legacy-system-documentation-archaeology.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "legacy-system-documentation-archaeology.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "legacy-system-documentation-archaeology.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "legacy-system-documentation-archaeology.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "legacy-system-documentation-archaeology.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "legacy-system-documentation-archaeology.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "legacy-system-documentation-archaeology.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "legacy-system-documentation-archaeology.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "logging-configuration-issues.md",
      "target": "buffer-overflow-vulnerabilities.md"
    },
    {
      "source": "logging-configuration-issues.md",
      "target": "excessive-logging.md"
    },
    {
      "source": "logging-configuration-issues.md",
      "target": "log-spam.md"
    },
    {
      "source": "logging-configuration-issues.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "logging-configuration-issues.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "logging-configuration-issues.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "long-lived-feature-branches.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "long-release-cycles.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "long-release-cycles.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "long-release-cycles.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "long-running-database-transactions.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "clever-code.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "release-instability.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "team-silos.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "maintenance-cost-increase.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "maintenance-cost-increase.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "maintenance-overhead.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "maintenance-overhead.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "maintenance-overhead.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "maintenance-overhead.md",
      "target": "procedural-background.md"
    },
    {
      "source": "maintenance-overhead.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "gold-plating.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "procedural-background.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "market-pressure.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "market-pressure.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "market-pressure.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "market-pressure.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "memory-barrier-inefficiency.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "memory-barrier-inefficiency.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "memory-barrier-inefficiency.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "memory-barrier-inefficiency.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "memory-barrier-inefficiency.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "memory-barrier-inefficiency.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "memory-barrier-inefficiency.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "memory-barrier-inefficiency.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "memory-barrier-inefficiency.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "memory-barrier-inefficiency.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "memory-barrier-inefficiency.md",
      "target": "race-conditions.md"
    },
    {
      "source": "memory-fragmentation.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "memory-fragmentation.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "memory-fragmentation.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "memory-fragmentation.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "buffer-overflow-vulnerabilities.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "log-spam.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "mental-fatigue.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "release-instability.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "system-outages.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "information-decay.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "user-confusion.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "context-switching-overhead.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "dma-coherency-issues.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "global-state-and-side-effects.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "gold-plating.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "inconsistent-codebase.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "large-feature-scope.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "long-release-cycles.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "procrastination-on-complex-tasks.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "race-conditions.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "release-instability.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "scope-creep.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "team-silos.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "user-confusion.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "user-confusion.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "vendor-dependency.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "shared-database.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "system-outages.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "user-confusion.md"
    },
    {
      "source": "modernization-roi-justification-failure.md",
      "target": "procedural-background.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "bloated-class.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "clever-code.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "database-schema-design-problems.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "hardcoded-values.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "high-turnover.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "information-decay.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "legacy-configuration-management-chaos.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "procedural-background.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "schema-evolution-paralysis.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "shared-database.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "team-coordination-issues.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "team-silos.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "user-confusion.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "bloated-class.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "procedural-background.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "shared-database.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "information-decay.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "log-spam.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "release-instability.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "system-outages.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "user-confusion.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "user-frustration.md"
    },
    {
      "source": "network-latency.md",
      "target": "endianness-conversion-overhead.md"
    },
    {
      "source": "network-latency.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "network-latency.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "network-latency.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "inappropriate-skillset.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "language-barriers.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "planning-credibility-issues.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "poor-communication.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "release-instability.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "team-silos.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "user-confusion.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "work-blocking.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "nitpicking-culture.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "nitpicking-culture.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "nitpicking-culture.md",
      "target": "reviewer-anxiety.md"
    },
    {
      "source": "nitpicking-culture.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "no-continuous-feedback-loop.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "no-continuous-feedback-loop.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "no-continuous-feedback-loop.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "no-continuous-feedback-loop.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "no-continuous-feedback-loop.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "no-continuous-feedback-loop.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "no-continuous-feedback-loop.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "no-continuous-feedback-loop.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "no-continuous-feedback-loop.md",
      "target": "procedural-background.md"
    },
    {
      "source": "no-continuous-feedback-loop.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "overworked-teams.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "perfectionist-review-culture.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "perfectionist-review-culture.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "perfectionist-review-culture.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "perfectionist-review-culture.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "perfectionist-review-culture.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "perfectionist-review-culture.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "planning-credibility-issues.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "cache-invalidation-problems.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "clever-code.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "database-schema-design-problems.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "defensive-coding-practices.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "dma-coherency-issues.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "excessive-object-allocation.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "high-connection-count.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "inadequate-requirements-gathering.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "inefficient-database-indexing.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "load-balancing-problems.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "log-spam.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "logging-configuration-issues.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "poor-planning.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "rate-limiting-issues.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "resource-waste.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "rest-api-design-issues.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "slow-database-queries.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "user-confusion.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "virtual-memory-thrashing.md"
    },
    {
      "source": "poor-communication.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "poor-contract-design.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "poor-contract-design.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "poor-contract-design.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "poor-contract-design.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "poor-contract-design.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "poor-contract-design.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "poor-contract-design.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "poor-contract-design.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "poor-contract-design.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "poor-contract-design.md",
      "target": "procedural-background.md"
    },
    {
      "source": "poor-contract-design.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "poor-contract-design.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "cache-invalidation-problems.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "database-schema-design-problems.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "shared-database.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "user-confusion.md"
    },
    {
      "source": "poor-interfaces-between-applications.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "poor-interfaces-between-applications.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "poor-interfaces-between-applications.md",
      "target": "duplicated-research-effort.md"
    },
    {
      "source": "poor-interfaces-between-applications.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "poor-interfaces-between-applications.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "poor-interfaces-between-applications.md",
      "target": "shared-database.md"
    },
    {
      "source": "poor-interfaces-between-applications.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "clever-code.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "logging-configuration-issues.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "procedural-background.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "user-confusion.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "communication-risk-outside-project.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "load-balancing-problems.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "poor-communication.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "power-struggles.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "quality-blind-spots.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "release-instability.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "resource-waste.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "individual-recognition-culture.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "information-decay.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "lock-contention.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "release-instability.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "scope-change-resistance.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "shared-database.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "system-outages.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "user-confusion.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "circular-dependency-problems.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "clever-code.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "defensive-coding-practices.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "difficult-to-test-code.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "fear-of-conflict.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "feature-gaps.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "hidden-dependencies.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "inadequate-code-reviews.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "inadequate-configuration-management.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "inadequate-error-handling.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "inadequate-initial-reviews.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "inadequate-integration-tests.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "inadequate-requirements-gathering.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "inadequate-test-infrastructure.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "inappropriate-skillset.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "inconsistent-coding-standards.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "increased-risk-of-bugs.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "inexperienced-developers.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "information-decay.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "information-fragmentation.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "insufficient-audit-logging.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "insufficient-code-review.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "invisible-nature-of-technical-debt.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "legacy-code-without-tests.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "logging-configuration-issues.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "lower-code-quality.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "nitpicking-culture.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "poor-contract-design.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "poor-domain-model.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "quality-blind-spots.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "quality-compromises.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "reduced-code-submission-frequency.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "reduced-review-participation.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "regression-bugs.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "review-process-avoidance.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "review-process-breakdown.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "reviewer-inexperience.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "ripple-effect-of-changes.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "skill-development-gaps.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "team-coordination-issues.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "test-debt.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "testing-environment-fragility.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "unpredictable-system-behavior.md"
    },
    {
      "source": "poorly-defined-responsibilities.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "power-struggles.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "premature-technology-introduction.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "procedural-programming-in-oop-languages.md",
      "target": "bloated-class.md"
    },
    {
      "source": "procedural-programming-in-oop-languages.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "procedural-programming-in-oop-languages.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "procedural-programming-in-oop-languages.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "procedural-programming-in-oop-languages.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "procedural-programming-in-oop-languages.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "procedural-programming-in-oop-languages.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "procedural-programming-in-oop-languages.md",
      "target": "procedural-background.md"
    },
    {
      "source": "procedural-programming-in-oop-languages.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "procrastination-on-complex-tasks.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "procrastination-on-complex-tasks.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "procrastination-on-complex-tasks.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "procrastination-on-complex-tasks.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "procrastination-on-complex-tasks.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "procrastination-on-complex-tasks.md",
      "target": "procedural-background.md"
    },
    {
      "source": "procrastination-on-complex-tasks.md",
      "target": "release-instability.md"
    },
    {
      "source": "procrastination-on-complex-tasks.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "procrastination-on-complex-tasks.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "procrastination-on-complex-tasks.md",
      "target": "user-confusion.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "changing-project-scope.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "feature-bloat.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "no-formal-change-control-process.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "power-struggles.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "race-conditions.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "release-instability.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "communication-risk-outside-project.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "dependency-on-supplier.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "development-disruption.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "knowledge-silos.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "quality-blind-spots.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "skill-development-gaps.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "team-silos.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "vendor-dependency.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "inadequate-onboarding.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "incorrect-index-type.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "index-fragmentation.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "inefficient-frontend-code.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "information-decay.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "poor-planning.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "poor-teamwork.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "procedural-background.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "release-instability.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "team-dysfunction.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "clever-code.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "feedback-isolation.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "no-continuous-feedback-loop.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "shared-database.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "queries-that-prevent-index-usage.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "queries-that-prevent-index-usage.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "queries-that-prevent-index-usage.md",
      "target": "graphql-complexity-issues.md"
    },
    {
      "source": "queries-that-prevent-index-usage.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "queries-that-prevent-index-usage.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "queries-that-prevent-index-usage.md",
      "target": "index-fragmentation.md"
    },
    {
      "source": "queries-that-prevent-index-usage.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "queries-that-prevent-index-usage.md",
      "target": "log-spam.md"
    },
    {
      "source": "queries-that-prevent-index-usage.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "queries-that-prevent-index-usage.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "queries-that-prevent-index-usage.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "queries-that-prevent-index-usage.md",
      "target": "unoptimized-file-access.md"
    },
    {
      "source": "rapid-team-growth.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "rapid-team-growth.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "rapid-team-growth.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "rapid-team-growth.md",
      "target": "gold-plating.md"
    },
    {
      "source": "rapid-team-growth.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "rate-limiting-issues.md",
      "target": "user-confusion.md"
    },
    {
      "source": "reduced-code-submission-frequency.md",
      "target": "procedural-background.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "development-disruption.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "gold-plating.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "increased-manual-work.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "information-decay.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "poor-communication.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "reduced-code-submission-frequency.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "release-instability.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "skill-development-gaps.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "system-outages.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "team-churn-impact.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "team-demoralization.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "user-confusion.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "user-frustration.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "increased-manual-work.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "knowledge-silos.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "large-feature-scope.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "clever-code.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "deadline-pressure.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "development-disruption.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "duplicated-work.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "extended-review-cycles.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "feature-bloat.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "gold-plating.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "high-turnover.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "implementation-starts-without-design.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "information-decay.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "integration-difficulties.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "knowledge-silos.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "large-feature-scope.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "mental-fatigue.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "procrastination-on-complex-tasks.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "release-instability.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "skill-development-gaps.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "slow-development-velocity.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "stagnant-architecture.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "stakeholder-frustration.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "team-demoralization.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "team-silos.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "technical-architecture-limitations.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "user-confusion.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "vendor-lock-in.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "reduced-review-participation.md",
      "target": "procedural-background.md"
    },
    {
      "source": "reduced-review-participation.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "reduced-team-flexibility.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "reduced-team-flexibility.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "changing-project-scope.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "competing-priorities.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "deadline-pressure.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "decision-avoidance.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "duplicated-work.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "feature-bloat.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "gold-plating.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "high-turnover.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "inadequate-integration-tests.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "increased-manual-work.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "individual-recognition-culture.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "inexperienced-developers.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "insufficient-worker-capacity.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "language-barriers.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "large-feature-scope.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "lower-code-quality.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "mental-fatigue.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "mentor-burnout.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "merge-conflicts.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "new-hire-frustration.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "planning-credibility-issues.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "priority-thrashing.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "procrastination-on-complex-tasks.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "reduced-code-submission-frequency.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "reduced-individual-productivity.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "reduced-review-participation.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "release-instability.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "resource-contention.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "skill-development-gaps.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "stakeholder-frustration.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "team-demoralization.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "team-silos.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "unclear-goals-and-priorities.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "unmotivated-employees.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "unproductive-meetings.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "user-confusion.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "user-frustration.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "abi-compatibility-issues.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "circular-dependency-problems.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "data-protection-risk.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "dependency-on-supplier.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "feature-creep.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "high-turnover.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "legacy-business-logic-extraction-difficulty.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "legacy-configuration-management-chaos.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "maintenance-cost-increase.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "obsolete-technologies.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "quality-blind-spots.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "review-process-avoidance.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "ripple-effect-of-changes.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "scope-creep.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "vendor-dependency.md"
    },
    {
      "source": "resource-contention.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "resource-contention.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "resource-contention.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "resource-contention.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "resource-contention.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "resource-contention.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "resource-contention.md",
      "target": "shared-database.md"
    },
    {
      "source": "resource-contention.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "review-bottlenecks.md",
      "target": "extended-review-cycles.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "clever-code.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "excessive-object-allocation.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "gold-plating.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "procedural-background.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "team-silos.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "test-debt.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "time-pressure.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "reviewer-inexperience.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "reviewer-inexperience.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "reviewer-inexperience.md",
      "target": "procedural-background.md"
    },
    {
      "source": "ripple-effect-of-changes.md",
      "target": "procedural-background.md"
    },
    {
      "source": "ripple-effect-of-changes.md",
      "target": "shared-database.md"
    },
    {
      "source": "ripple-effect-of-changes.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "ripple-effect-of-changes.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "scaling-inefficiencies.md",
      "target": "shared-database.md"
    },
    {
      "source": "scaling-inefficiencies.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "scope-change-resistance.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "scope-change-resistance.md",
      "target": "gold-plating.md"
    },
    {
      "source": "scope-change-resistance.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "scope-creep.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "scope-creep.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "scope-creep.md",
      "target": "slow-development-velocity.md"
    },
    {
      "source": "scope-creep.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "second-system-effect.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "second-system-effect.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "second-system-effect.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "second-system-effect.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "second-system-effect.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "secret-management-problems.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "secret-management-problems.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "secret-management-problems.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "secret-management-problems.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "secret-management-problems.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "single-entry-point-design.md",
      "target": "shared-database.md"
    },
    {
      "source": "single-entry-point-design.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "changing-project-scope.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "clever-code.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "difficult-developer-onboarding.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "duplicated-work.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "extended-review-cycles.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "implementation-starts-without-design.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "increased-error-rates.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "inefficient-frontend-code.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "inexperienced-developers.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "insufficient-worker-capacity.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "maintenance-cost-increase.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "merge-conflicts.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "n-plus-one-query-problem.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "no-formal-change-control-process.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "procrastination-on-complex-tasks.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "release-instability.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "team-silos.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "procedural-background.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "development-disruption.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "gold-plating.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "information-decay.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "language-barriers.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "mental-fatigue.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "obsolete-technologies.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "poor-communication.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "priority-thrashing.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "procedural-background.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "release-instability.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "team-demoralization.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "team-silos.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "user-confusion.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "user-frustration.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "gold-plating.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "release-instability.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "system-outages.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "team-demoralization.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "user-confusion.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "user-frustration.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "release-instability.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "user-confusion.md"
    },
    {
      "source": "strangler-fig-pattern-failures.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "strangler-fig-pattern-failures.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "superficial-code-reviews.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "superficial-code-reviews.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "superficial-code-reviews.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "superficial-code-reviews.md",
      "target": "procedural-background.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "shared-database.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "system-outages.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "user-confusion.md"
    },
    {
      "source": "team-churn-impact.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "team-confusion.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "team-confusion.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "team-confusion.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "team-confusion.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "team-confusion.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "team-confusion.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "team-confusion.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "bloated-class.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "clever-code.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "individual-recognition-culture.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "log-spam.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "procedural-background.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "release-instability.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "user-confusion.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "individual-recognition-culture.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "information-decay.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "lock-contention.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "release-instability.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "scope-change-resistance.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "shared-database.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "system-outages.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "user-confusion.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "clever-code.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "high-turnover.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "inability-to-innovate.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "information-decay.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "procedural-background.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "reduced-code-submission-frequency.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "release-instability.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "team-demoralization.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "team-silos.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "user-confusion.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "complex-implementation-paths.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "data-migration-integrity-issues.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "dependency-on-supplier.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "hidden-dependencies.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "implementation-rework.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "implementation-starts-without-design.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "legacy-configuration-management-chaos.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "maintenance-cost-increase.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "quality-blind-spots.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "rate-limiting-issues.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "resource-waste.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "serialization-deserialization-bottlenecks.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "skill-development-gaps.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "unbounded-data-growth.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "vendor-dependency.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "clever-code.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "individual-recognition-culture.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "procedural-background.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "shared-database.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "user-confusion.md"
    },
    {
      "source": "technology-stack-fragmentation.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "technology-stack-fragmentation.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "technology-stack-fragmentation.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "test-debt.md",
      "target": "bloated-class.md"
    },
    {
      "source": "test-debt.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "test-debt.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "test-debt.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "test-debt.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "test-debt.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "test-debt.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "test-debt.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "test-debt.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "test-debt.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "test-debt.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "test-debt.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "test-debt.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "test-debt.md",
      "target": "shared-database.md"
    },
    {
      "source": "test-debt.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "test-debt.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "test-debt.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "test-debt.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "test-debt.md",
      "target": "user-confusion.md"
    },
    {
      "source": "testing-environment-fragility.md",
      "target": "procedural-background.md"
    },
    {
      "source": "testing-environment-fragility.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "procedural-background.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "shared-database.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "buffer-overflow-vulnerabilities.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "high-resource-utilization-on-client.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "high-turnover.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "improper-event-listener-management.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "increased-error-rates.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "integer-overflow-underflow.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "long-running-database-transactions.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "maintenance-cost-increase.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "release-instability.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "second-system-effect.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "shared-database.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "bloated-class.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "circular-references.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "excessive-object-allocation.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "graphql-complexity-issues.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "high-client-side-resource-consumption.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "improper-event-listener-management.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "index-fragmentation.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "insufficient-worker-capacity.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "n-plus-one-query-problem.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "resource-contention.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "scope-creep.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "serialization-deserialization-bottlenecks.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "shared-database.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "stack-overflow-errors.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "unreleased-resources.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "poor-communication.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "system-outages.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "user-confusion.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "changing-project-scope.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "clever-code.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "user-confusion.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "information-decay.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "poor-communication.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "procedural-background.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "team-silos.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "user-confusion.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "gold-plating.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "high-turnover.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "language-barriers.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "procrastination-on-complex-tasks.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "skill-development-gaps.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "team-silos.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "user-confusion.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "information-decay.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "user-confusion.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "feature-creep.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "global-state-and-side-effects.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "procedural-background.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "unproductive-meetings.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "unproductive-meetings.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "unproductive-meetings.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "unproductive-meetings.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "unproductive-meetings.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "unproductive-meetings.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "unrealistic-schedule.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "unrealistic-schedule.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "unrealistic-schedule.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "unrealistic-schedule.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "unrealistic-schedule.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "unrealistic-schedule.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "unrealistic-schedule.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "unrealistic-schedule.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "alignment-and-padding-issues.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "bloated-class.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "buffer-overflow-vulnerabilities.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "cache-invalidation-problems.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "circular-dependency-problems.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "communication-risk-outside-project.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "competing-priorities.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "database-query-performance-issues.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "development-disruption.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "excessive-logging.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "excessive-object-allocation.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "global-state-and-side-effects.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "high-client-side-resource-consumption.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "high-connection-count.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "high-resource-utilization-on-client.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "high-turnover.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "increased-error-rates.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "index-fragmentation.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "inefficient-frontend-code.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "information-decay.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "insufficient-worker-capacity.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "legacy-configuration-management-chaos.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "load-balancing-problems.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "log-spam.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "long-running-database-transactions.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "n-plus-one-query-problem.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "null-pointer-dereferences.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "obsolete-technologies.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "poor-planning.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "release-instability.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "resource-contention.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "resource-waste.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "scope-creep.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "second-system-effect.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "service-timeouts.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "session-management-issues.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "shared-database.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "stack-overflow-errors.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "strangler-fig-pattern-failures.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "system-outages.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "test-debt.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "unused-indexes.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "user-confusion.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "user-frustration.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "user-trust-erosion.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "virtual-memory-thrashing.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "unused-indexes.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "unused-indexes.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "unused-indexes.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "unused-indexes.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "unused-indexes.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "unused-indexes.md",
      "target": "shared-database.md"
    },
    {
      "source": "unused-indexes.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "cascade-delays.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "circular-dependency-problems.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "context-switching-overhead.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "database-query-performance-issues.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "dependency-version-conflicts.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "excessive-logging.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "feature-creep.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "graphql-complexity-issues.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "high-client-side-resource-consumption.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "high-connection-count.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "high-resource-utilization-on-client.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "inadequate-error-handling.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "inadequate-integration-tests.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "increased-error-rates.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "index-fragmentation.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "inefficient-database-indexing.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "inefficient-frontend-code.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "information-decay.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "insufficient-worker-capacity.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "integration-difficulties.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "large-feature-scope.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "load-balancing-problems.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "log-spam.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "long-running-database-transactions.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "microservice-communication-overhead.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "n-plus-one-query-problem.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "network-latency.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "rate-limiting-issues.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "reduced-code-submission-frequency.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "resource-contention.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "rest-api-design-issues.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "serialization-deserialization-bottlenecks.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "service-timeouts.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "slow-database-queries.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "stack-overflow-errors.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "strangler-fig-pattern-failures.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "system-outages.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "team-coordination-issues.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "time-pressure.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "unbounded-data-growth.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "unoptimized-file-access.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "user-confusion.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "user-frustration.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "vendor-dependency-entrapment.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "vendor-dependency-entrapment.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "vendor-dependency-entrapment.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "vendor-dependency-entrapment.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "vendor-dependency-entrapment.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "vendor-dependency-entrapment.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "vendor-dependency-entrapment.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "vendor-dependency-entrapment.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "vendor-dependency-entrapment.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "dependency-on-supplier.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "user-confusion.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "work-blocking.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "work-blocking.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "work-blocking.md",
      "target": "deadlock-conditions.md"
    }
  ]
};

const width = Math.min(1200, window.innerWidth * 0.9);
const height = window.innerHeight - 60; // Account for header

const categoryColors = {
    // 15 core categories with vibrant, distinct colors
    'Architecture': '#3498db',    // Blue
    'Business': '#e74c3c',        // Red
    'Code': '#f39c12',            // Orange
    'Communication': '#9b59b6',   // Purple
    'Culture': '#e67e22',         // Dark Orange
    'Database': '#2ecc71',        // Green
    'Dependencies': '#16a085',    // Teal
    'Management': '#e91e63',      // Pink
    'Operations': '#34495e',      // Dark Blue-Gray
    'Performance': '#f1c40f',     // Yellow
    'Process': '#27ae60',         // Dark Green
    'Requirements': '#8e44ad',    // Dark Purple
    'Security': '#c0392b',        // Dark Red
    'Team': '#1abc9c',            // Turquoise
    'Testing': '#ff6b35'          // Red-Orange
};

const color = (category) => categoryColors[category] || '#6c757d';

const simulation = d3.forceSimulation(graph.nodes)
    .force("link", d3.forceLink(graph.links).id(d => d.id).distance(100))
    .force("charge", d3.forceManyBody().strength(-1000))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("x", d3.forceX(width / 2).strength(0.1))
    .force("y", d3.forceY(height / 2).strength(0.1))
    .force("collision", d3.forceCollide().radius(30));

const svg = d3.select("#visualization-container").append("svg")
    .attr("width", width)
    .attr("height", height)
    .call(d3.zoom()
        .extent([[0, 0], [width, height]])
        .scaleExtent([0.1, 8])
        .on("zoom", function (event) {
            container.attr("transform", event.transform);
        }));

// Container for all elements that should be zoomed/panned
const container = svg.append("g");

// Arrowhead marker with dynamic position
svg.append('defs').append('marker')
    .attr('id', 'arrowhead')
    .attr('viewBox', '-0 -5 10 10')
    .attr('refX', 35)
    .attr('refY', 0)
    .attr('orient', 'auto')
    .attr('markerWidth', 8)
    .attr('markerHeight', 8)
    .attr('xoverflow', 'visible')
    .append('svg:path')
    .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
    .attr('fill', '#999')
    .style('stroke', 'none');

const link = container.append('g')
    .attr('class', 'link')
    .selectAll('line')
    .data(graph.links)
    .enter().append('line')
    .attr('stroke-width', function (d) { return Math.sqrt(d.value || 1); })
    .attr('marker-end', 'url(#arrowhead)');

const node = container.append('g')
    .attr('class', 'nodes')
    .selectAll('g')
    .data(graph.nodes)
    .enter().append('g')
    .attr('class', 'node')
    .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

// Variables to handle selection and opacity
let selectedNode = null;
const nonActiveOpacity = 0.2;

node.append('circle')
    .attr('r', function (d) { return d.size || 10; })
    .attr('fill', function (d) { return color(d.category); });

node.append('text')
    .text(function (d) { return d.title; })
    .attr('dy', -15);

const tooltip = d3.select(".tooltip");

// Update event handlers and add 'click' handler.
node.on("mouseover", (event, d) => {
    tooltip.transition().style("opacity", 1);
    tooltip.html(`<strong>${d.title}</strong><br/><em>${d.category}</em><br/>${d.description}`)
        .style("left", (event.pageX + 15) + "px")
        .style("top", (event.pageY - 28) + "px");
})
    .on("mouseout", () => {
        tooltip.transition().style("opacity", 0);
    })
    .on("click", (event, d) => {
        event.stopPropagation(); // Prevent click from propagating to background (SVG)
        // If clicking the already selected node, deselect. Otherwise, select.
        selectedNode = (selectedNode && selectedNode.id === d.id) ? null : d;
        if (pathManager.isCreatingPath) {
            selectedNode = d;
            if (pathManager.isAddingNode) {
                pathManager.currentPath.add(selectedNode.id);
            }
            else if (pathManager.isRemovingNode) {
                pathManager.currentPath.delete(selectedNode.id);
            }
        }
        updateStyles();
    })
    .on("dblclick", (event, d) => {
        event.stopPropagation();
        openModal(d);
    });

// Click on background to deselect all
svg.on('click', function () {
    if (!pathManager.isCreatingPath) { //To "disable" background click when creating a path
        selectedNode = null;
        updateStyles();
    }
});

const categories = [...new Set(graph.nodes.map(d => d.category))].sort();
const legend = d3.select(".legend");

categories.forEach(category => {
    const legendItem = legend.append("div").attr("class", "legend-item");
    legendItem.append("div")
        .attr("class", "legend-color")
        .style("background-color", color(category));
    legendItem.append("span").text(category);
});

simulation
    .on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("transform", d => `translate(${d.x}, ${d.y})`);
    });

// Central function to update node and link opacity
function updateStyles() {
    if (!pathManager.isCreatingPath) {
        node.style('opacity', 1);
        link.style('stroke-opacity', 0.6)
            .style('stroke', '#adb5bd')
            .style('filter', 'none');

        if (selectedNode) {
            // If a node is selected
            node.style('opacity', d => (d.id === selectedNode.id ? 1 : nonActiveOpacity));
            link.style('stroke-opacity', l => (l.source.id === selectedNode.id || l.target.id === selectedNode.id) ? 1 : nonActiveOpacity)
                .style('stroke', l => (l.source.id === selectedNode.id || l.target.id === selectedNode.id) ? '#6c757d' : '#adb5bd');
        }
        return;
    }

    if (!selectedNode) return;

    node.select('circle')
        // Highlight selected node in green
        // Highlight nodes in path in black
        // Rest in white
        .style('stroke', d => {
            if (d.id === selectedNode.id) return 'green';
            else if (pathManager.currentPath.has(d.id)) return 'black';
            else return '#fff';
        })
        //Set stroke-width to 3 to all nodes in path, otherwise 1.5.
        .style('stroke-width', d => { pathManager.currentPath.has(d.id) || d.id === selectedNode.id ? 3 : 1.5 });

    // Enable to select/add nodes in path or neighbors of selected node.
    //const neighboringNodes = getNeighboringNodes(selectedNode.id);
    //node.style('pointer-events', d => { (pathManager.currentPath.has(d.id) || neighboringNodes.has(d.id) || d.id === selectedNode.id) ? 'all' : 'none'; });

    // Set opacity 1 to all nodes in path.
    node.style('opacity', d => pathManager.currentPath.has(d.id) ? 1 : nonActiveOpacity);

    // Set stroke color and opacity black to the links connected to the selected node.
    // Set Stroke color lime to all links among the nodes in path.
    link.style('filter', l => (pathManager.currentPath.has(l.source.id) && pathManager.currentPath.has(l.target.id)) ? 'drop-shadow(0px 0px 3px lime)' : 'none')
        .style('stroke-opacity', l => {
            const inPath = pathManager.currentPath.has(l.source.id) && pathManager.currentPath.has(l.target.id);
            const connectedToSelected = l.source.id === selectedNode.id || l.target.id === selectedNode.id;
            return inPath || connectedToSelected ? 1 : nonActiveOpacity; // Full opacity
        })
        .style('stroke', l => {
            const inPath = pathManager.currentPath.has(l.source.id) && pathManager.currentPath.has(l.target.id);
            const connectedToSelected = l.source.id === selectedNode.id || l.target.id === selectedNode.id;
            return inPath ? 'lime' : (connectedToSelected ? 'black' : '#adb5bd'); // Lime for path, black for connected, gray otherwise
        });
}

function dragstarted(event, d) {
    if (!event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
}

function dragged(event, d) {
    d.fx = event.x;
    d.fy = event.y;
}

function dragended(event, d) {
    if (!event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
}

// Create Path Object Literal
const pathManager = {
    _isCreatingPath: false,
    _isAddingNode: false,
    _isRemovingNode: false,
    _storageKey: 'allMyPaths',
    currentPath: new Set(),
    selectedNode: null,

    get isCreatingPath() {
        return this._isCreatingPath;
    },
    set isCreatingPath(val) {
        this._isCreatingPath = val;
        if (!val) {
            this._isAddingNode = false;
            this._isRemovingNode = false;
        }
    },
    get isAddingNode() {
        return this._isAddingNode;
    },
    set isAddingNode(val) {
        if (this._isCreatingPath) {
            this._isAddingNode = val;
            if (val) this._isRemovingNode = false;
        } else {
            this._isAddingNode = false;
        }
    },
    get isRemovingNode() {
        return this._isRemovingNode;
    },
    set isRemovingNode(val) {
        if (this._isCreatingPath) {
            this._isRemovingNode = val;
            if (val) this._isAddingNode = false;
        } else {
            this._isRemovingNode = false;
        }
    },
    get localStoragePaths() {
        const pathsString = localStorage.getItem(this._storageKey);
        return pathsString ? JSON.parse(pathsString) : [];
    },
    set localStoragePaths(pathsArray) {
        localStorage.setItem(this._storageKey, JSON.stringify(pathsArray));
    },
    /**
     * Add a new path to the existing array in localStorage.
     * @param {object} newPath - The new path to add.
     */
    addPathToLocalStorage: function (newPath) {
        const currentPaths = this.localStoragePaths;
        currentPaths.push(newPath);
        this.localStoragePaths = currentPaths;
    },
    /**
     * Remove a path by its name.
     * @param {string} pathNameToRemove - The name of the path to remove.
     */
    removePathFromLocalStorage: function (pathNameToRemove) {
        let currentPaths = this.localStoragePaths;
        currentPaths = currentPaths.filter(path => path.name !== pathNameToRemove);
        this.localStoragePaths = currentPaths;
    },
    outCreateMode: function () {
        this.isCreatingPath = false;
        selectedNode = null;
        this.currentPath.clear();
        updateStyles();
    },
    getPathfromLocalStorage: function (pathName) {
        const currentPaths = this.localStoragePaths;
        return currentPaths.filter(path => path.name === pathName)[0];
    },
    updatePathsInDom: function () {
        const paths = this.localStoragePaths;
        pathListContainer.innerHTML = ''; // Clear existing paths
        if (paths.length === 0) {
            pathListContainer.innerHTML = '<em>No saved paths.</em>';
            return;
        }
        paths.forEach(path => {
            const pathItem = document.createElement('div');
            pathItem.className = 'control-item';
            pathItem.innerHTML = `
                <span class="control-icon">📍</span>
                <span class="control-text" data-path-name="${path.name}">${path.name}</span>
                <span class="delete-path-btn" title="Delete Path" data-path-name="${path.name}">🚫</span>
                <span class="export-path-btn" title="Export" data-path-name="${path.name}">📥</span>
            `;
            pathListContainer.appendChild(pathItem);
        });
    },
    exportPath: function (pathName) {
        const path = this.getPathfromLocalStorage(pathName);
        const blob = new Blob([JSON.stringify(path, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${pathName}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    },
    async importPath() {
        try {
            const file = await new Promise(resolve => {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = '.json';
                input.onchange = () => resolve(input.files[0]);
                input.click();
            });

            if (!file) return;

            const fileContent = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = event => resolve(event.target.result);
                reader.onerror = error => reject(error);
                reader.readAsText(file);
            });

            this.addPathToLocalStorage(JSON.parse(fileContent));
            this.updatePathsInDom();

        } catch (error) {
            console.error("Import failed:", error);
            alert("The file could not be imported. Make sure it is a valid JSON.");
        }
    }
};

// HTML UI Event Listeners
const createPathButton = document.querySelector('#create-path-btn');
const importPathButton = document.querySelector('#import-path-btn');
const pathCreateOnTooltip = document.querySelector('.path-creator-on-tooltip');
const addNodeButton = document.querySelector('#add-node-btn');
const removeNodeButton = document.querySelector('#remove-node-btn');
const savePathButton = document.querySelector('#save-path-btn');
const cancelPathButton = document.querySelector('#cancel-path-btn');
const pathListContainer = document.querySelector('#path-list-container');

createPathButton.addEventListener('click', () => {
    createPathButton.disabled = true;
    addNodeButton.disabled = true;
    pathManager.isCreatingPath = true;
    pathManager.isAddingNode = true;
    pathManager.currentPath.clear();
    selectedNode = null;
    updateStyles();
    pathCreateOnTooltip.style.visibility = 'visible';
});
importPathButton.addEventListener('click', () => {
    pathManager.importPath();
});
addNodeButton.addEventListener('click', () => {
    pathManager.isAddingNode = true;
    addNodeButton.disabled = true;
    removeNodeButton.disabled = false;
    pathCreateOnTooltip.style.backgroundColor = 'lime'; // Light green background
});
removeNodeButton.addEventListener('click', () => {
    pathManager.isRemovingNode = true;
    removeNodeButton.disabled = true;
    addNodeButton.disabled = false;
    pathCreateOnTooltip.style.backgroundColor = '#fa858fff'; // Light red background
});
savePathButton.addEventListener('click', () => {
    if (pathManager.currentPath.size < 2) {
        alert('Please select at least two nodes to create a path.');
        return;
    }
    const pathArray = Array.from(pathManager.currentPath);
    const pathName = prompt('Enter a name for the new path:');
    if (pathName) {
        // Save the path (e.g., send to server or store locally)
        pathManager.addPathToLocalStorage({ name: pathName, nodes: pathArray });
        // Update the UI to show the new path
        pathManager.updatePathsInDom();
    }
});
cancelPathButton.addEventListener('click', () => {
    pathManager.outCreateMode();
    pathCreateOnTooltip.style.visibility = 'hidden';
    createPathButton.disabled = false;
});
pathListContainer.addEventListener('click', (event) => {
    // Check if a DELETE BUTTON was clicked
    const deleteButton = event.target.closest('.delete-path-btn');
    if (deleteButton) {
        const pathNameToDelete = deleteButton.dataset.pathName;
        pathManager.removePathFromLocalStorage(pathNameToDelete);
        pathManager.updatePathsInDom();
        return;
    }

    // Check if a PATH NAME was clicked
    const pathNameLink = event.target.closest('.control-text');
    if (pathNameLink) {
        event.preventDefault();
        const pathNameToLoad = pathNameLink.dataset.pathName;
        console.log(`Loading path: ${pathNameToLoad}`);

        const nodesInPath = pathManager.getPathfromLocalStorage(pathNameToLoad).nodes;

        pathManager.currentPath = new Set(nodesInPath);
        createPathButton.disabled = true;
        addNodeButton.disabled = true;
        pathManager.isCreatingPath = true;
        pathManager.isAddingNode = true;
        pathCreateOnTooltip.style.visibility = 'visible';
        selectedNode = nodesInPath[0];
        updateStyles();
        return
    }
    // Check if an EXPORT BUTTON was clicked
    const exportButton = event.target.closest('.export-path-btn');
    if (exportButton) {
        const pathNameToExport = exportButton.dataset.pathName;
        pathManager.exportPath(pathNameToExport);
        return;
    }
});

document.addEventListener('DOMContentLoaded', () => {
    //Load existing paths from localStorage and display them
    pathManager.updatePathsInDom();
});

// Modal functionality
const modal = document.getElementById('modal');
const modalTitle = document.getElementById('modal-title');
const modalCategory = document.getElementById('modal-category');
const modalBody = document.getElementById('modal-body');
const modalClose = document.getElementById('modal-close');

function openModal(nodeData) {
    modalTitle.textContent = nodeData.title;
    modalCategory.textContent = nodeData.category;
    modalBody.innerHTML = '<div class="loading">Loading problem details...</div>';
    modal.style.display = 'block';

    // Update URL hash when opening modal
    window.history.replaceState(null, null, '#' + nodeData.id);

    // Fetch the generated HTML file content
    const problemPath = nodeData.id.replace('.md', '.html');
    fetch(`problems/${problemPath}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.text();
        })
        .then(content => {
            displayHtmlContent(content, nodeData);
        })
        .catch(error => {
            console.error('Error fetching problem content:', error);
            modalBody.innerHTML = `
                <div class="error">
                    <h3>Error Loading Content</h3>
                    <p>Could not load the full problem description.</p>
                    <div class="modal-description">${nodeData.description}</div>
                </div>
            `;
        });
}

function displayHtmlContent(content, nodeData) {
    // Parse the Jekyll-generated HTML to extract the main content
    const parser = new DOMParser();
    const doc = parser.parseFromString(content, 'text/html');

    // Find the main content area - Jekyll typically puts content in .page-content or main
    let mainContent = doc.querySelector('.page-content main') ||
        doc.querySelector('main') ||
        doc.querySelector('.page-content') ||
        doc.querySelector('article');

    if (!mainContent) {
        // Fallback: try to find content by looking for common patterns
        mainContent = doc.querySelector('div[class*="content"]') || doc.body;
    }

    let html = '';
    if (mainContent) {
        html = mainContent.innerHTML;

        // Convert relative markdown links to problem-link spans for modal navigation
        html = html.replace(/href="([^"]*\.md)"/g, function (match, mdFile) {
            const filename = mdFile.split('/').pop();
            return `class="problem-link" data-problem="${filename}"`;
        });

        // Also handle relative problem links
        html = html.replace(/href="\/problems\/([^"]*)\.html"/g, function (match, problemName) {
            return `class="problem-link" data-problem="${problemName}.md"`;
        });

        // Remove any Jekyll-specific elements we don't want
        html = html.replace(/<div class="page-header">.*?<\/div>/gs, '');
    } else {
        html = '<p>Could not extract content from the page.</p>';
    }

    modalBody.innerHTML = `
        ${html}
    `;

    // Add click handlers for problem links
    const problemLinks = modalBody.querySelectorAll('.problem-link');
    problemLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const problemFile = this.getAttribute('data-problem');
            const targetNode = graph.nodes.find(n => n.id === problemFile);
            if (targetNode) {
                openModal(targetNode);
            }
        });
    });

    // Convert regular links to open in new tab
    const externalLinks = modalBody.querySelectorAll('a:not(.problem-link)');
    externalLinks.forEach(link => {
        link.setAttribute('target', '_blank');
    });
}

function closeModal() {
    modal.style.display = 'none';
    // Clear URL hash when closing modal
    window.history.replaceState(null, null, window.location.pathname);
}

// Event listeners
modalClose.onclick = closeModal;
window.onclick = function (event) {
    if (event.target === modal) {
        closeModal();
    }
};

// Escape key to close modal
document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && modal.style.display === 'block') {
        closeModal();
    }
});

// Function to highlight and center a specific node
function focusNode(nodeId) {
    const targetNode = graph.nodes.find(n => n.id === nodeId);
    if (!targetNode) return false;

    // Center the node
    const scale = d3.zoomTransform(svg.node()).k || 1;
    const x = -targetNode.x * scale + width / 2;
    const y = -targetNode.y * scale + height / 2;

    svg.transition()
        .duration(750)
        .call(d3.zoom().transform, d3.zoomIdentity.translate(x, y).scale(scale));

    // Highlight the node temporarily
    const nodeElement = node.filter(d => d.id === nodeId).select('circle');
    const originalColor = nodeElement.attr('fill');

    nodeElement
        .transition()
        .duration(200)
        .attr('fill', '#ff6b6b')
        .attr('r', function (d) { return (d.size || 10) * 1.5; })
        .transition()
        .duration(1000)
        .attr('fill', originalColor)
        .attr('r', function (d) { return d.size || 10; });

    return true;
}

// Handle URL fragments on page load
function handleUrlFragment() {
    const hash = window.location.hash.substring(1); // Remove the #
    if (hash) {
        // Wait for the simulation to settle before focusing
        setTimeout(() => {
            if (!focusNode(hash)) {
                console.warn('Node not found for hash:', hash);
            }
        }, 1000);
    }
}

// Initialize URL fragment handling after simulation starts
simulation.on('end', handleUrlFragment);

// Also handle hash changes during navigation
window.addEventListener('hashchange', handleUrlFragment);