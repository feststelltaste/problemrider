const graph = {
  "nodes": [
    {
      "id": "shared-dependencies.md",
      "title": "Shared Dependencies",
      "description": "A situation where multiple components or services share a common set of libraries and frameworks.",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "n-plus-one-query-problem.md",
      "title": "N+1 Query Problem",
      "description": "An application makes numerous unnecessary database calls to fetch related data where a single, more efficient query would suffice, causing significant performance degradation.",
      "category": "Database",
      "size": 18
    },
    {
      "id": "high-api-latency.md",
      "title": "High API Latency",
      "description": "The time it takes for an API to respond to a request is excessively long, leading to poor application performance and a negative user experience.",
      "category": "Performance",
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
      "id": "logging-configuration-issues.md",
      "title": "Logging Configuration Issues",
      "description": "Improper logging configuration results in missing critical information, excessive log volume, or security vulnerabilities.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "increased-bug-count.md",
      "title": "Increased Bug Count",
      "description": "Changes introduce new defects more frequently, leading to a higher defect rate in production and degraded software quality.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "increased-technical-shortcuts.md",
      "title": "Increased Technical Shortcuts",
      "description": "Pressure to deliver leads to more quick fixes and workarounds instead of proper solutions, creating future maintenance problems.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "testing-complexity.md",
      "title": "Testing Complexity",
      "description": "Quality assurance must verify the same functionality in multiple locations, which increases the testing effort and the risk of missing bugs.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "database-schema-design-problems.md",
      "title": "Database Schema Design Problems",
      "description": "Poor database schema design creates performance issues, data integrity problems, and maintenance difficulties.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "secret-management-problems.md",
      "title": "Secret Management Problems",
      "description": "Inadequate handling of sensitive credentials and secrets creates security vulnerabilities and operational challenges.",
      "category": "Operations",
      "size": 19
    },
    {
      "id": "feedback-isolation.md",
      "title": "Feedback Isolation",
      "description": "Development teams operate without regular input from stakeholders and users, leading to products that miss requirements and user needs.",
      "category": "Business",
      "size": 19
    },
    {
      "id": "reduced-individual-productivity.md",
      "title": "Reduced Individual Productivity",
      "description": "Individual developers complete fewer tasks and take longer to resolve problems despite maintaining the same work effort.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "ripple-effect-of-changes.md",
      "title": "Ripple Effect of Changes",
      "description": "A small change in one part of the system requires modifications in many other seemingly unrelated parts, indicating high coupling.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "legacy-api-versioning-nightmare.md",
      "title": "Legacy API Versioning Nightmare",
      "description": "Legacy systems with poorly designed APIs create versioning and backward compatibility challenges that compound over time",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "development-disruption.md",
      "title": "Development Disruption",
      "description": "The development team is constantly interrupted by urgent production issues, which disrupts planned work and reduces overall productivity.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "incomplete-projects.md",
      "title": "Incomplete Projects",
      "description": "Features are started but never finished due to shifting priorities, leading to a great deal of wasted effort and a sense of frustration for the development team.",
      "category": "Process",
      "size": 18
    },
    {
      "id": "implicit-knowledge.md",
      "title": "Implicit Knowledge",
      "description": "Critical system knowledge exists as unwritten assumptions, tribal knowledge, and undocumented practices rather than being explicitly captured.",
      "category": "Communication",
      "size": 17
    },
    {
      "id": "information-decay.md",
      "title": "Information Decay",
      "description": "System documentation becomes outdated, inaccurate, or incomplete over time, making it unreliable for decision-making and system understanding.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "deadline-pressure.md",
      "title": "Deadline Pressure",
      "description": "Intense pressure to meet deadlines leads to rushed decisions, shortcuts, and compromised quality in software development.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "deployment-coupling.md",
      "title": "Deployment Coupling",
      "description": "A situation where multiple components or services must be deployed together, even if only one of them has changed.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "high-database-resource-utilization.md",
      "title": "High Database Resource Utilization",
      "description": "The database server consistently operates with high CPU or memory usage, risking instability and slowing down all dependent services.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "extended-cycle-times.md",
      "title": "Extended Cycle Times",
      "description": "The time from when work begins until it's completed and delivered becomes much longer than the actual work time required.",
      "category": "Process",
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
      "id": "long-running-transactions.md",
      "title": "Long-Running Transactions",
      "description": "Database transactions that remain open for a long time can hold locks, consume resources, and block other operations.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "maintenance-paralysis.md",
      "title": "Maintenance Paralysis",
      "description": "Teams avoid necessary improvements because they cannot verify that changes don't break existing functionality.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "high-coupling-low-cohesion.md",
      "title": "High Coupling and Low Cohesion",
      "description": "Software components are overly dependent on each other and perform too many unrelated functions, making the system difficult to change and understand.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "bloated-class.md",
      "title": "Bloated Class",
      "description": "A class that has grown so large that it has become difficult to understand, maintain, and test.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "priority-thrashing.md",
      "title": "Priority Thrashing",
      "description": "Work priorities change frequently and unexpectedly, causing constant task switching and disrupting planned work flow.",
      "category": "Process",
      "size": 19
    },
    {
      "id": "monitoring-gaps.md",
      "title": "Monitoring Gaps",
      "description": "Insufficient production monitoring and observability make it difficult to detect and diagnose issues in a timely manner, leading to longer outages and more severe consequences.",
      "category": "Code",
      "size": 16
    },
    {
      "id": "poor-user-experience-ux-design.md",
      "title": "Poor User Experience (UX) Design",
      "description": "The application is difficult to use, confusing, or does not meet user needs.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "session-management-issues.md",
      "title": "Session Management Issues",
      "description": "Poor session handling creates security vulnerabilities through session hijacking, fixation, or improper lifecycle management.",
      "category": "Security",
      "size": 17
    },
    {
      "id": "extended-research-time.md",
      "title": "Extended Research Time",
      "description": "Developers spend significant portions of their day researching rather than implementing, due to knowledge gaps or complex legacy systems.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "vendor-lock-in.md",
      "title": "Vendor Lock-In",
      "description": "System is overly dependent on a specific vendor's tools or APIs, limiting future options",
      "category": "Code",
      "size": 16
    },
    {
      "id": "unbounded-data-structures.md",
      "title": "Unbounded Data Structures",
      "description": "Data structures that grow indefinitely without proper pruning or size limits, leading to memory exhaustion and performance degradation.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "workaround-culture.md",
      "title": "Workaround Culture",
      "description": "Teams implement increasingly complex workarounds rather than fixing root issues, creating layers of technical debt.",
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
      "id": "resource-allocation-failures.md",
      "title": "Resource Allocation Failures",
      "description": "Objects, connections, file handles, or other system resources are allocated but never properly deallocated or closed, leading to resource exhaustion.",
      "category": "Code",
      "size": 16
    },
    {
      "id": "limited-team-learning.md",
      "title": "Limited Team Learning",
      "description": "A situation where a team does not learn from its mistakes and does not improve over time.",
      "category": "Process",
      "size": 19
    },
    {
      "id": "reduced-review-participation.md",
      "title": "Reduced Review Participation",
      "description": "Many team members avoid participating in code reviews, concentrating review burden on a few individuals and reducing coverage.",
      "category": "Process",
      "size": 18
    },
    {
      "id": "graphql-complexity-issues.md",
      "title": "GraphQL Complexity Issues",
      "description": "GraphQL queries become too complex or expensive to execute, causing performance problems and potential denial-of-service vulnerabilities.",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "vendor-dependency-entrapment.md",
      "title": "Vendor Dependency Entrapment",
      "description": "Legacy systems become trapped by discontinued vendor products, forcing expensive custom support contracts or complete system replacement",
      "category": "Code",
      "size": 15
    },
    {
      "id": "copy-paste-programming.md",
      "title": "Copy-Paste Programming",
      "description": "Developers frequently copy and paste code rather than creating reusable components, leading to maintenance nightmares and subtle bugs.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "extended-review-cycles.md",
      "title": "Extended Review Cycles",
      "description": "Code reviews require multiple rounds of feedback and revision, significantly extending the time from code submission to approval.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "reduced-code-submission-frequency.md",
      "title": "Reduced Code Submission Frequency",
      "description": "Developers batch multiple changes together or delay submissions to avoid frequent code review cycles, reducing feedback quality and integration frequency.",
      "category": "Process",
      "size": 17
    },
    {
      "id": "communication-risk-outside-project.md",
      "title": "Communication Risk Outside Project",
      "description": "External stakeholders are left uninformed, leading to surprises and misaligned expectations about project progress.",
      "category": "Communication",
      "size": 18
    },
    {
      "id": "uneven-work-flow.md",
      "title": "Uneven Work Flow",
      "description": "Work progresses in irregular fits and starts rather than flowing smoothly through the development process.",
      "category": "Process",
      "size": 19
    },
    {
      "id": "refactoring-avoidance.md",
      "title": "Refactoring Avoidance",
      "description": "The development team actively avoids refactoring the codebase, even when they acknowledge it's necessary, due to fear of introducing new bugs.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "procedural-programming-in-oop-languages.md",
      "title": "Procedural Programming in OOP Languages",
      "description": "Code is written in a procedural style within object-oriented languages, leading to large, monolithic functions and poor encapsulation.",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "inappropriate-skillset.md",
      "title": "Inappropriate Skillset",
      "description": "Team members lack essential knowledge or experience needed for their assigned roles and responsibilities.",
      "category": "Culture",
      "size": 19
    },
    {
      "id": "duplicated-research-effort.md",
      "title": "Duplicated Research Effort",
      "description": "Multiple team members research the same topics independently, wasting time and failing to build collective knowledge.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "inconsistent-codebase.md",
      "title": "Inconsistent Codebase",
      "description": "The project's code lacks uniform style, coding standards, and design patterns, making it difficult to read, maintain, and onboard new developers.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "long-build-and-test-times.md",
      "title": "Long Build and Test Times",
      "description": "A situation where it takes a long time to build and test a system.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "single-points-of-failure.md",
      "title": "Single Points of Failure",
      "description": "Progress is blocked when specific knowledge holders or system components are unavailable, creating critical dependencies.",
      "category": "Management",
      "size": 19
    },
    {
      "id": "inadequate-error-handling.md",
      "title": "Inadequate Error Handling",
      "description": "Poor error handling mechanisms fail to gracefully manage exceptions, leading to application crashes and poor user experiences.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "planning-credibility-issues.md",
      "title": "Planning Credibility Issues",
      "description": "Future estimates and plans are questioned or ignored due to history of inaccurate predictions and missed commitments.",
      "category": "Business",
      "size": 18
    },
    {
      "id": "missing-rollback-strategy.md",
      "title": "Missing Rollback Strategy",
      "description": "There's no tested method to undo a deployment if things go wrong, increasing risk",
      "category": "Code",
      "size": 16
    },
    {
      "id": "integration-difficulties.md",
      "title": "Integration Difficulties",
      "description": "Connecting with modern services requires extensive workarounds due to architectural limitations or outdated integration patterns.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "difficult-to-test-code.md",
      "title": "Difficult to Test Code",
      "description": "Components cannot be easily tested in isolation due to tight coupling, global dependencies, or complex setup requirements.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "high-number-of-database-queries.md",
      "title": "High Number of Database Queries",
      "description": "A single user request triggers an unexpectedly large number of database queries, leading to performance degradation and increased database load.",
      "category": "Database",
      "size": 17
    },
    {
      "id": "excessive-disk-io.md",
      "title": "Excessive Disk I/O",
      "description": "The system performs a high number of disk read/write operations, indicating inefficient data access or processing.",
      "category": "Performance",
      "size": 19
    },
    {
      "id": "code-duplication.md",
      "title": "Code Duplication",
      "description": "Similar or identical code exists in multiple places, making maintenance difficult and introducing inconsistency risks.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "poor-encapsulation.md",
      "title": "Poor Encapsulation",
      "description": "Data and the behavior that acts on that data are not bundled together in a single, cohesive unit, leading to a lack of data hiding and a high degree of coupling.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "user-trust-erosion.md",
      "title": "User Trust Erosion",
      "description": "Frequent issues and emergency fixes damage user confidence in the system's reliability, leading to a decline in user engagement and satisfaction.",
      "category": "Business",
      "size": 18
    },
    {
      "id": "null-pointer-dereferences.md",
      "title": "Null Pointer Dereferences",
      "description": "Programs attempt to access memory through null or invalid pointers, causing crashes and potential security vulnerabilities.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "incorrect-index-type.md",
      "title": "Incorrect Index Type",
      "description": "Using an inappropriate type of database index for a given query pattern, leading to inefficient data retrieval.",
      "category": "Performance",
      "size": 17
    },
    {
      "id": "assumption-based-development.md",
      "title": "Assumption-Based Development",
      "description": "Developers make decisions based on assumptions about requirements or user needs rather than validating their understanding.",
      "category": "Communication",
      "size": 19
    },
    {
      "id": "perfectionist-culture.md",
      "title": "Perfectionist Culture",
      "description": "A culture of perfectionism and a reluctance to release anything that is not 100% perfect can lead to analysis paralysis and long release cycles.",
      "category": "Culture",
      "size": 19
    },
    {
      "id": "scope-change-resistance.md",
      "title": "Scope Change Resistance",
      "description": "Necessary changes to project scope are avoided or resisted due to process constraints, contract limitations, or organizational inertia.",
      "category": "Management",
      "size": 17
    },
    {
      "id": "review-process-breakdown.md",
      "title": "Review Process Breakdown",
      "description": "Code review practices fail to identify critical issues, provide meaningful feedback, or improve code quality due to systemic process failures.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "single-entry-point-design.md",
      "title": "Single Entry Point Design",
      "description": "A design where all requests to a system must go through a single object or component.",
      "category": "Architecture",
      "size": 16
    },
    {
      "id": "increased-manual-testing-effort.md",
      "title": "Increased Manual Testing Effort",
      "description": "A disproportionate amount of time is spent on manual testing due to a lack of automation.",
      "category": "Process",
      "size": 19
    },
    {
      "id": "breaking-changes.md",
      "title": "Breaking Changes",
      "description": "API updates break existing client integrations, causing compatibility issues and forcing costly emergency fixes.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "deployment-environment-inconsistencies.md",
      "title": "Deployment Environment Inconsistencies",
      "description": "Differences between deployment environments cause applications to behave differently or fail when moved between environments.",
      "category": "Operations",
      "size": 19
    },
    {
      "id": "garbage-collection-pressure.md",
      "title": "Garbage Collection Pressure",
      "description": "Excessive object allocation and deallocation causes frequent garbage collection cycles, creating performance pauses and reducing application throughput.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "atomic-operation-overhead.md",
      "title": "Atomic Operation Overhead",
      "description": "Excessive use of atomic operations creates performance bottlenecks due to memory synchronization overhead and cache coherency traffic.",
      "category": "Architecture",
      "size": 16
    },
    {
      "id": "high-resource-utilization-on-client.md",
      "title": "High Resource Utilization on Client",
      "description": "Client applications may consume excessive CPU or memory, leading to a poor user experience, especially on less powerful devices.",
      "category": "Performance",
      "size": 15
    },
    {
      "id": "dependency-version-conflicts.md",
      "title": "Dependency Version Conflicts",
      "description": "Conflicting versions of dependencies cause runtime errors, build failures, and unexpected behavior in applications.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "no-formal-change-control-process.md",
      "title": "No Formal Change Control Process",
      "description": "Changes to project scope or requirements are not formally evaluated or approved, leading to uncontrolled scope creep and project delays.",
      "category": "Process",
      "size": 19
    },
    {
      "id": "cargo-culting.md",
      "title": "Cargo Culting",
      "description": "Uncritical adoption of technical solutions without understanding their underlying principles and context",
      "category": "Architecture",
      "size": 17
    },
    {
      "id": "maintenance-overhead.md",
      "title": "Maintenance Overhead",
      "description": "A disproportionate amount of time and effort is spent on maintaining the existing system, often due to duplicated code and a lack of reusable components.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "tool-limitations.md",
      "title": "Tool Limitations",
      "description": "Inadequate development tools slow down common tasks and reduce developer productivity and satisfaction.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "circular-dependency-problems.md",
      "title": "Circular Dependency Problems",
      "description": "Components depend on each other in circular patterns, creating initialization issues, testing difficulties, and architectural complexity.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "inadequate-onboarding.md",
      "title": "Inadequate Onboarding",
      "description": "New users struggle to understand how to use the application, leading to early abandonment.",
      "category": "Business",
      "size": 17
    },
    {
      "id": "reviewer-anxiety.md",
      "title": "Reviewer Anxiety",
      "description": "Team members feel uncertain and anxious about conducting code reviews, leading to avoidance or superficial review practices.",
      "category": "Culture",
      "size": 18
    },
    {
      "id": "debugging-difficulties.md",
      "title": "Debugging Difficulties",
      "description": "Finding and fixing bugs becomes challenging due to complex code architecture, poor logging, or inadequate development tools.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "cascade-failures.md",
      "title": "Cascade Failures",
      "description": "A single change triggers a chain reaction of failures across multiple system components.",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "hardcoded-values.md",
      "title": "Hardcoded Values",
      "description": "Magic numbers and fixed strings reduce flexibility, making configuration and adaptation difficult",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "poor-naming-conventions.md",
      "title": "Poor Naming Conventions",
      "description": "Variables, functions, classes, and other code elements are named in ways that don't clearly communicate their purpose or meaning.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "review-bottlenecks.md",
      "title": "Review Bottlenecks",
      "description": "The code review process becomes a significant bottleneck, delaying the delivery of new features and bug fixes.",
      "category": "Process",
      "size": 19
    },
    {
      "id": "inefficient-development-environment.md",
      "title": "Inefficient Development Environment",
      "description": "The team is slowed down by a slow and cumbersome development environment",
      "category": "Code",
      "size": 18
    },
    {
      "id": "lock-contention.md",
      "title": "Lock Contention",
      "description": "Multiple threads compete for the same locks, causing threads to block and reducing parallel execution efficiency.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "release-anxiety.md",
      "title": "Release Anxiety",
      "description": "The development team is anxious and stressed about deployments due to the high risk of failure and the pressure to get it right.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "misaligned-deliverables.md",
      "title": "Misaligned Deliverables",
      "description": "The delivered product or feature does not match the expectations or requirements of the stakeholders.",
      "category": "Communication",
      "size": 19
    },
    {
      "id": "maintenance-cost-increase.md",
      "title": "Maintenance Cost Increase",
      "description": "The resources required to maintain, support, and update software systems grow over time, consuming increasing portions of development budgets.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "power-struggles.md",
      "title": "Power Struggles",
      "description": "Internal conflicts between departments or managers interfere with decision-making and project progress.",
      "category": "Management",
      "size": 19
    },
    {
      "id": "conflicting-reviewer-opinions.md",
      "title": "Conflicting Reviewer Opinions",
      "description": "Multiple reviewers provide contradictory guidance on the same code changes, creating confusion and inefficiency.",
      "category": "Communication",
      "size": 19
    },
    {
      "id": "load-balancing-problems.md",
      "title": "Load Balancing Problems",
      "description": "Load balancing mechanisms distribute traffic inefficiently or fail to adapt to changing conditions, causing performance issues and service instability.",
      "category": "Operations",
      "size": 18
    },
    {
      "id": "operational-overhead.md",
      "title": "Operational Overhead",
      "description": "A significant amount of time and resources are spent on emergency response and firefighting, rather than on planned development and innovation.",
      "category": "Code",
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
      "id": "legacy-system-documentation-archaeology.md",
      "title": "Legacy System Documentation Archaeology",
      "description": "Critical system knowledge exists only in obsolete documentation formats, outdated diagrams, and departed employees' tribal knowledge",
      "category": "Communication",
      "size": 19
    },
    {
      "id": "delayed-issue-resolution.md",
      "title": "Delayed Issue Resolution",
      "description": "Problems persist longer because no one feels responsible for fixing them, leading to accumulated technical debt and user frustration.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "accumulated-decision-debt.md",
      "title": "Accumulated Decision Debt",
      "description": "Deferred decisions create compound complexity for future choices, making the system increasingly difficult to evolve.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "stakeholder-dissatisfaction.md",
      "title": "Stakeholder Dissatisfaction",
      "description": "Business stakeholders become unhappy with project outcomes, development speed, or communication quality.",
      "category": "Communication",
      "size": 19
    },
    {
      "id": "merge-conflicts.md",
      "title": "Merge Conflicts",
      "description": "Multiple developers frequently modify the same large functions or files, creating version control conflicts that slow development.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "complex-domain-model.md",
      "title": "Complex Domain Model",
      "description": "The business domain being modeled in software is inherently complex, making the system difficult to understand and implement correctly.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "constantly-shifting-deadlines.md",
      "title": "Constantly Shifting Deadlines",
      "description": "The project's end date is repeatedly pushed back to accommodate new feature requests, leading to a loss of credibility and a great deal of frustration for the development team.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "increased-risk-of-bugs.md",
      "title": "Increased Risk of Bugs",
      "description": "Code complexity and lack of clarity make it more likely that developers will introduce defects when making changes.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "inadequate-test-data-management.md",
      "title": "Inadequate Test Data Management",
      "description": "The use of unrealistic, outdated, or insufficient test data leads to tests that do not accurately reflect real-world scenarios.",
      "category": "Code",
      "size": 16
    },
    {
      "id": "inconsistent-behavior.md",
      "title": "Inconsistent Behavior",
      "description": "The same business process produces different outcomes depending on where it's triggered, leading to a confusing and unpredictable user experience.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "obsolete-technologies.md",
      "title": "Obsolete Technologies",
      "description": "The system relies on outdated tools, frameworks, or languages that make modern development practices difficult to implement.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "organizational-structure-mismatch.md",
      "title": "Organizational Structure Mismatch",
      "description": "A situation where the structure of the organization does not match the architecture of the system.",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "system-outages.md",
      "title": "System Outages",
      "description": "Service interruptions and system failures occur frequently, causing business disruption and user frustration.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "shadow-systems.md",
      "title": "Shadow Systems",
      "description": "Alternative solutions developed outside official channels undermine standardization and create hidden dependencies.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "inexperienced-developers.md",
      "title": "Inexperienced Developers",
      "description": "Development team lacks the knowledge and experience to implement best practices and maintainable solutions.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "slow-incident-resolution.md",
      "title": "Slow Incident Resolution",
      "description": "Problems and outages take excessive time to diagnose and resolve, prolonging business impact and user frustration.",
      "category": "Operations",
      "size": 19
    },
    {
      "id": "legacy-skill-shortage.md",
      "title": "Legacy Skill Shortage",
      "description": "Critical shortage of developers with knowledge of legacy technologies creates bottlenecks and single points of failure for system maintenance",
      "category": "Management",
      "size": 19
    },
    {
      "id": "configuration-drift.md",
      "title": "Configuration Drift",
      "description": "System configurations gradually diverge from intended standards over time, creating inconsistencies and reliability issues.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "team-churn-impact.md",
      "title": "Team Churn Impact",
      "description": "Over time, as developers join and leave the team, they bring inconsistent practices and knowledge gaps that degrade code quality.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "change-management-chaos.md",
      "title": "Change Management Chaos",
      "description": "Changes to systems occur without coordination, oversight, or impact assessment, leading to conflicts and unintended consequences.",
      "category": "Management",
      "size": 17
    },
    {
      "id": "gradual-performance-degradation.md",
      "title": "Gradual Performance Degradation",
      "description": "Application performance slowly deteriorates over time due to resource leaks, accumulating technical debt, or inefficient algorithms.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "high-connection-count.md",
      "title": "High Connection Count",
      "description": "A large number of open database connections, even if idle, can consume significant memory resources and lead to connection rejections.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "poor-system-environment.md",
      "title": "Poor System Environment",
      "description": "The system is deployed in an unstable, misconfigured, or unsuitable environment that causes outages, performance issues, and operational difficulties.",
      "category": "Operations",
      "size": 19
    },
    {
      "id": "rushed-approvals.md",
      "title": "Rushed Approvals",
      "description": "Pull requests are approved quickly without thorough examination due to time pressure or process issues.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "monolithic-architecture-constraints.md",
      "title": "Monolithic Architecture Constraints",
      "description": "Large monolithic codebases become difficult to maintain, scale, and deploy as they grow in size and complexity.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "poor-interfaces-between-applications.md",
      "title": "Poor Interfaces Between Applications",
      "description": "Disconnected or poorly defined interfaces lead to fragile integrations and inconsistent data",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "perfectionist-review-culture.md",
      "title": "Perfectionist Review Culture",
      "description": "Team culture emphasizes making code perfect through reviews rather than focusing on meaningful improvements, leading to excessive revision cycles.",
      "category": "Culture",
      "size": 19
    },
    {
      "id": "insecure-data-transmission.md",
      "title": "Insecure Data Transmission",
      "description": "Sensitive data transmitted without proper encryption or security controls, exposing it to interception and unauthorized access.",
      "category": "Security",
      "size": 18
    },
    {
      "id": "competing-priorities.md",
      "title": "Competing Priorities",
      "description": "Multiple urgent projects or initiatives compete for the same limited resources, creating conflicts and inefficiencies.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "user-confusion.md",
      "title": "User Confusion",
      "description": "End users encounter different behavior for what should be identical operations, leading to frustration and a loss of trust in the system.",
      "category": "Requirements",
      "size": 18
    },
    {
      "id": "fear-of-failure.md",
      "title": "Fear of Failure",
      "description": "A pervasive fear of making mistakes or failing can lead to inaction, risk aversion, and a reluctance to innovate within a development team.",
      "category": "Culture",
      "size": 20
    },
    {
      "id": "test-debt.md",
      "title": "Test Debt",
      "description": "The accumulated risk from inadequate or neglected quality assurance, leading to a fragile product and slow development velocity.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "password-security-weaknesses.md",
      "title": "Password Security Weaknesses",
      "description": "Weak password policies, inadequate storage mechanisms, and poor authentication practices create security vulnerabilities.",
      "category": "Security",
      "size": 18
    },
    {
      "id": "unrealistic-deadlines.md",
      "title": "Unrealistic Deadlines",
      "description": "Management sets aggressive deadlines that do not account for the actual effort required, leading to compromised quality and unsustainable work practices.",
      "category": "Management",
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
      "id": "lazy-loading.md",
      "title": "Lazy Loading",
      "description": "The use of lazy loading in an ORM framework leads to a large number of unnecessary database queries, which can significantly degrade application performance.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "capacity-mismatch.md",
      "title": "Capacity Mismatch",
      "description": "Available capacity at different stages of the development process doesn't match demand patterns, creating bottlenecks and underutilization.",
      "category": "Performance",
      "size": 18
    },
    {
      "id": "vendor-relationship-strain.md",
      "title": "Vendor Relationship Strain",
      "description": "Tensions and conflicts arise between the organization and external vendors due to misaligned expectations, poor communication, or contract issues.",
      "category": "Communication",
      "size": 17
    },
    {
      "id": "insufficient-testing.md",
      "title": "Insufficient Testing",
      "description": "The testing process is not comprehensive enough, leading to a high defect rate in production.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "invisible-nature-of-technical-debt.md",
      "title": "Invisible Nature of Technical Debt",
      "description": "The impact of technical debt is not visible to non-technical stakeholders, making it hard to justify addressing it and allocate resources for improvement.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "knowledge-gaps.md",
      "title": "Knowledge Gaps",
      "description": "Lack of understanding about systems, business requirements, or technical domains leads to extended research time and suboptimal solutions.",
      "category": "Communication",
      "size": 19
    },
    {
      "id": "wasted-development-effort.md",
      "title": "Wasted Development Effort",
      "description": "Significant development work is abandoned, reworked, or becomes obsolete due to poor planning, changing requirements, or inefficient processes.",
      "category": "Performance",
      "size": 20
    },
    {
      "id": "excessive-class-size.md",
      "title": "Excessive Class Size",
      "description": "Classes become overly large and complex, making them difficult to understand, maintain, and test.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "user-frustration.md",
      "title": "User Frustration",
      "description": "Users become dissatisfied with system reliability, usability, or performance, leading to decreased adoption and negative feedback.",
      "category": "Business",
      "size": 19
    },
    {
      "id": "thread-pool-exhaustion.md",
      "title": "Thread Pool Exhaustion",
      "description": "All available threads in the thread pool are consumed by long-running or blocked operations, preventing new tasks from being processed.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "poor-domain-model.md",
      "title": "Poor Domain Model",
      "description": "Core business concepts are poorly understood or reflected in the system, leading to fragile logic and miscommunication",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "lower-code-quality.md",
      "title": "Lower Code Quality",
      "description": "Burned-out or rushed developers are more likely to make mistakes, leading to an increase in defects.",
      "category": "Code",
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
      "id": "communication-risk-within-project.md",
      "title": "Communication Risk Within Project",
      "description": "Misunderstandings and unclear messages reduce coordination and trust among project team members.",
      "category": "Communication",
      "size": 19
    },
    {
      "id": "vendor-dependency.md",
      "title": "Vendor Dependency",
      "description": "Excessive reliance on external vendors or suppliers creates risks when they become unavailable, change terms, or fail to meet expectations.",
      "category": "Dependencies",
      "size": 17
    },
    {
      "id": "reduced-team-flexibility.md",
      "title": "Reduced Team Flexibility",
      "description": "The team's ability to adapt to changing requirements, reassign work, or respond to unexpected challenges is significantly limited.",
      "category": "Dependencies",
      "size": 16
    },
    {
      "id": "data-protection-risk.md",
      "title": "Data Protection Risk",
      "description": "Handling of personal or sensitive data lacks safeguards, exposing the project to legal and ethical issues",
      "category": "Process",
      "size": 17
    },
    {
      "id": "inadequate-test-infrastructure.md",
      "title": "Inadequate Test Infrastructure",
      "description": "Missing tools, environments, or automation make thorough testing slow or impossible",
      "category": "Code",
      "size": 18
    },
    {
      "id": "upstream-timeouts.md",
      "title": "Upstream Timeouts",
      "description": "Services that consume an API fail because they do not receive a response within their configured timeout window.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "project-authority-vacuum.md",
      "title": "Project Authority Vacuum",
      "description": "Critical projects lack sufficient organizational backing and executive sponsorship to overcome resistance and secure resources.",
      "category": "Management",
      "size": 17
    },
    {
      "id": "poor-operational-concept.md",
      "title": "Poor Operational Concept",
      "description": "Lack of planning for monitoring, maintenance, or support leads to post-launch instability",
      "category": "Operations",
      "size": 19
    },
    {
      "id": "poor-planning.md",
      "title": "Poor Planning",
      "description": "Teams do not have clear plans or realistic estimates of the work involved, leading to project delays and resource allocation problems.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "duplicated-effort.md",
      "title": "Duplicated Effort",
      "description": "Multiple team members unknowingly work on the same problems or implement similar solutions independently.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "convenience-driven-development.md",
      "title": "Convenience-Driven Development",
      "description": "A development practice where developers choose the easiest and most convenient solution, rather than the best solution.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "undefined-code-style-guidelines.md",
      "title": "Undefined Code Style Guidelines",
      "description": "The team lacks clear, agreed-upon coding standards, resulting in subjective stylistic feedback and inconsistent code.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "synchronization-problems.md",
      "title": "Synchronization Problems",
      "description": "Updates to one copy of duplicated logic don't get applied to other copies, causing divergent behavior across the system.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "increased-manual-work.md",
      "title": "Increased Manual Work",
      "description": "Developers spend time on repetitive tasks that should be automated, reducing time available for actual development work.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "micromanagement-culture.md",
      "title": "Micromanagement Culture",
      "description": "Management culture that requires approval for routine decisions, reducing team autonomy and creating unnecessary bottlenecks.",
      "category": "Culture",
      "size": 18
    },
    {
      "id": "avoidance-behaviors.md",
      "title": "Avoidance Behaviors",
      "description": "Complex tasks are postponed or avoided entirely due to cognitive overload, fear, or perceived difficulty.",
      "category": "Management",
      "size": 19
    },
    {
      "id": "knowledge-dependency.md",
      "title": "Knowledge Dependency",
      "description": "Team members remain dependent on specific experienced individuals for knowledge and decision-making longer than appropriate for their role and tenure.",
      "category": "Communication",
      "size": 18
    },
    {
      "id": "unbounded-data-growth.md",
      "title": "Unbounded Data Growth",
      "description": "Data structures, caches, or databases grow indefinitely without proper pruning, size limits, or archiving strategies.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "resource-waste.md",
      "title": "Resource Waste",
      "description": "Available resources are not utilized effectively, leading to underutilization while other areas remain constrained or overloaded.",
      "category": "Performance",
      "size": 15
    },
    {
      "id": "cross-system-data-synchronization-problems.md",
      "title": "Cross-System Data Synchronization Problems",
      "description": "Maintaining data consistency between legacy and modern systems during migration creates complex synchronization challenges and potential data corruption",
      "category": "Code",
      "size": 19
    },
    {
      "id": "unused-indexes.md",
      "title": "Unused Indexes",
      "description": "The database has indexes that are never used by any queries, which still consume storage space and add overhead to write operations.",
      "category": "Code",
      "size": 15
    },
    {
      "id": "team-members-not-engaged-in-review-process.md",
      "title": "Team Members Not Engaged in Review Process",
      "description": "Code reviews are often assigned to the same people, or reviewers do not provide meaningful feedback, leading to a bottleneck and reduced quality.",
      "category": "Communication",
      "size": 18
    },
    {
      "id": "procrastination-on-complex-tasks.md",
      "title": "Procrastination on Complex Tasks",
      "description": "Difficult or cognitively demanding work is consistently postponed in favor of easier, more immediately gratifying tasks.",
      "category": "Culture",
      "size": 16
    },
    {
      "id": "data-migration-complexities.md",
      "title": "Data Migration Complexities",
      "description": "Complex data migration processes create risks of data loss, corruption, or extended downtime during system updates.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "mixed-coding-styles.md",
      "title": "Mixed Coding Styles",
      "description": "A situation where different parts of the codebase use different formatting, naming conventions, and design patterns.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "resource-contention.md",
      "title": "Resource Contention",
      "description": "The server is overloaded, and the application is competing for limited resources like CPU, memory, or I/O.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "long-running-database-transactions.md",
      "title": "Long-Running Database Transactions",
      "description": "Database transactions remain open for extended periods, holding locks and consuming resources, which can block other operations.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "past-negative-experiences.md",
      "title": "Past Negative Experiences",
      "description": "A situation where developers are hesitant to make changes to the codebase because of negative experiences in the past.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "increased-stress-and-burnout.md",
      "title": "Increased Stress and Burnout",
      "description": "Team members are overworked and stressed due to unrealistic expectations and time pressure, leading to decreased morale and productivity.",
      "category": "Culture",
      "size": 19
    },
    {
      "id": "abi-compatibility-issues.md",
      "title": "ABI Compatibility Issues",
      "description": "Application Binary Interface incompatibilities between different versions of libraries or system components cause runtime failures or undefined behavior.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "analysis-paralysis.md",
      "title": "Analysis Paralysis",
      "description": "Teams become stuck in research phases without moving to implementation, preventing actual progress on development work.",
      "category": "Management",
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
      "id": "defensive-coding-practices.md",
      "title": "Defensive Coding Practices",
      "description": "Developers write overly verbose code, excessive comments, or unnecessary defensive logic to preempt anticipated criticism during code reviews.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "constant-firefighting.md",
      "title": "Constant Firefighting",
      "description": "The development team is perpetually occupied with fixing bugs and addressing urgent issues, leaving little to no time for new feature development.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "misconfigured-connection-pools.md",
      "title": "Misconfigured Connection Pools",
      "description": "Application connection pools are improperly set up, leading to inefficient resource utilization or connection exhaustion.",
      "category": "Code",
      "size": 16
    },
    {
      "id": "stakeholder-frustration.md",
      "title": "Stakeholder Frustration",
      "description": "Business stakeholders become frustrated with development progress, quality, or communication, leading to strained relationships and reduced support.",
      "category": "Business",
      "size": 18
    },
    {
      "id": "premature-technology-introduction.md",
      "title": "Premature Technology Introduction",
      "description": "New frameworks, tools, or platforms are introduced without proper evaluation, adding risk and learning overhead to projects.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "hidden-dependencies.md",
      "title": "Hidden Dependencies",
      "description": "Workarounds and patches create unexpected dependencies between system components that are not obvious from the code structure.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "bikeshedding.md",
      "title": "Bikeshedding",
      "description": "Reviewers focus on trivial issues like whitespace and variable names instead of more important issues like logic and design.",
      "category": "Process",
      "size": 18
    },
    {
      "id": "superficial-code-reviews.md",
      "title": "Superficial Code Reviews",
      "description": "Code reviews focus only on surface-level issues like formatting and style while missing important design, logic, or security problems.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "code-review-inefficiency.md",
      "title": "Code Review Inefficiency",
      "description": "The code review process takes excessive time, provides limited value, or creates bottlenecks in the development workflow.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "competitive-disadvantage.md",
      "title": "Competitive Disadvantage",
      "description": "Users switch to competitors who offer better experience or more features due to technical shortcomings.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "rapid-team-growth.md",
      "title": "Rapid Team Growth",
      "description": "Teams expand in size quickly without adequate preparation, overwhelming existing infrastructure and support systems.",
      "category": "Management",
      "size": 18
    },
    {
      "id": "bottleneck-formation.md",
      "title": "Bottleneck Formation",
      "description": "Specific team members, processes, or system components become constraints that limit the overall flow and productivity of development work.",
      "category": "Performance",
      "size": 19
    },
    {
      "id": "review-process-avoidance.md",
      "title": "Review Process Avoidance",
      "description": "Team members actively seek ways to bypass or minimize code review requirements, undermining the quality assurance process.",
      "category": "Process",
      "size": 18
    },
    {
      "id": "communication-breakdown.md",
      "title": "Communication Breakdown",
      "description": "Team members fail to effectively share information, coordinate work, or collaborate, leading to duplicated effort and misaligned solutions.",
      "category": "Communication",
      "size": 19
    },
    {
      "id": "modernization-strategy-paralysis.md",
      "title": "Modernization Strategy Paralysis",
      "description": "Teams become overwhelmed by modernization options (rewrite, refactor, replace, retire) and fail to make decisions, leaving systems in limbo",
      "category": "Management",
      "size": 19
    },
    {
      "id": "insufficient-worker-capacity.md",
      "title": "Insufficient Worker Capacity",
      "description": "There are not enough worker processes or threads to handle the incoming volume of tasks in an asynchronous system, leading to growing queues.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "unoptimized-file-access.md",
      "title": "Unoptimized File Access",
      "description": "Applications read or write files inefficiently, leading to excessive disk I/O and slow performance.",
      "category": "Performance",
      "size": 16
    },
    {
      "id": "difficult-code-reuse.md",
      "title": "Difficult Code Reuse",
      "description": "It is difficult to reuse code in different contexts because it is not designed in a modular and reusable way.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "brittle-codebase.md",
      "title": "Brittle Codebase",
      "description": "The existing code is difficult to modify without introducing new bugs, making maintenance and feature development risky.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "history-of-failed-changes.md",
      "title": "History of Failed Changes",
      "description": "A past record of failed deployments or changes creates a culture of fear and resistance to future modifications.",
      "category": "Culture",
      "size": 19
    },
    {
      "id": "log-injection-vulnerabilities.md",
      "title": "Log Injection Vulnerabilities",
      "description": "Unsanitized user input in log messages allows attackers to inject malicious content that can compromise log integrity or exploit log processing systems.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "slow-database-queries.md",
      "title": "Slow Database Queries",
      "description": "Application performance degrades due to inefficient data retrieval from the database.",
      "category": "Performance",
      "size": 17
    },
    {
      "id": "inconsistent-knowledge-acquisition.md",
      "title": "Inconsistent Knowledge Acquisition",
      "description": "New team members learn different aspects and depths of system knowledge depending on their mentor or learning path, creating uneven skill distribution.",
      "category": "Communication",
      "size": 18
    },
    {
      "id": "poor-documentation.md",
      "title": "Poor Documentation",
      "description": "System documentation is outdated, incomplete, inaccurate, or difficult to find and use effectively.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "configuration-chaos.md",
      "title": "Configuration Chaos",
      "description": "System configurations are inconsistent, difficult to manage, and prone to drift, causing unpredictable behavior across environments.",
      "category": "Operations",
      "size": 19
    },
    {
      "id": "planning-dysfunction.md",
      "title": "Planning Dysfunction",
      "description": "Project planning processes fail to create realistic timelines, allocate resources effectively, or account for project complexities and risks.",
      "category": "Process",
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
      "id": "slow-development-velocity.md",
      "title": "Slow Development Velocity",
      "description": "The team consistently fails to deliver features and bug fixes at a predictable and acceptable pace, with overall productivity systematically declining.",
      "category": "Business",
      "size": 19
    },
    {
      "id": "unclear-sharing-expectations.md",
      "title": "Unclear Sharing Expectations",
      "description": "A situation where it is not clear what information should be shared with the rest of the team.",
      "category": "Process",
      "size": 17
    },
    {
      "id": "partial-bug-fixes.md",
      "title": "Partial Bug Fixes",
      "description": "Issues appear to be resolved but resurface in different contexts because the fix was not applied to all instances of the duplicated code.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "mental-fatigue.md",
      "title": "Mental Fatigue",
      "description": "Developers report feeling exhausted and mentally drained without accomplishing significant work, often due to cognitive overhead and inefficient workflows.",
      "category": "Culture",
      "size": 19
    },
    {
      "id": "inability-to-innovate.md",
      "title": "Inability to Innovate",
      "description": "The team is so bogged down in day-to-day maintenance tasks that they have no time to think about future improvements or new approaches.",
      "category": "Management",
      "size": 18
    },
    {
      "id": "language-barriers.md",
      "title": "Language Barriers",
      "description": "Differences in language or terminology hinder smooth communication and understanding within the team.",
      "category": "Communication",
      "size": 17
    },
    {
      "id": "style-arguments-in-code-reviews.md",
      "title": "Style Arguments in Code Reviews",
      "description": "A situation where a significant amount of time in code reviews is spent debating trivial style issues instead of focusing on logic and design.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "approval-dependencies.md",
      "title": "Approval Dependencies",
      "description": "Work progress is frequently blocked by the need for approvals from specific individuals, creating bottlenecks and delays.",
      "category": "Dependencies",
      "size": 20
    },
    {
      "id": "feature-bloat.md",
      "title": "Feature Bloat",
      "description": "Products become overly complex with numerous features that dilute the core value proposition and confuse users.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "complex-implementation-paths.md",
      "title": "Complex Implementation Paths",
      "description": "Simple business requirements require complex technical solutions due to architectural constraints or design limitations.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "over-reliance-on-utility-classes.md",
      "title": "Over-Reliance on Utility Classes",
      "description": "The excessive use of utility classes with static methods can lead to a procedural style of programming and a lack of proper object-oriented design.",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "inefficient-code.md",
      "title": "Inefficient Code",
      "description": "The code responsible for handling a request is computationally expensive or contains performance bottlenecks.",
      "category": "Performance",
      "size": 18
    },
    {
      "id": "inadequate-integration-tests.md",
      "title": "Inadequate Integration Tests",
      "description": "The interactions between different modules or services are not thoroughly tested, leading to integration failures.",
      "category": "Architecture",
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
      "id": "duplicated-work.md",
      "title": "Duplicated Work",
      "description": "Multiple team members unknowingly work on the same tasks or solve the same problems, leading to wasted effort and potential conflicts.",
      "category": "Communication",
      "size": 19
    },
    {
      "id": "architectural-mismatch.md",
      "title": "Architectural Mismatch",
      "description": "New business requirements don't fit well within existing architectural constraints, requiring extensive workarounds or compromises.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "inconsistent-coding-standards.md",
      "title": "Inconsistent Coding Standards",
      "description": "Lack of uniform coding standards across the codebase creates maintenance difficulties and reduces code readability and quality.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "delayed-decision-making.md",
      "title": "Delayed Decision Making",
      "description": "Important decisions that affect development progress are postponed or take excessive time to make, creating bottlenecks and uncertainty.",
      "category": "Management",
      "size": 19
    },
    {
      "id": "quality-blind-spots.md",
      "title": "Quality Blind Spots",
      "description": "Critical system behaviors and failure modes remain undetected due to gaps in testing coverage and verification practices.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "tangled-cross-cutting-concerns.md",
      "title": "Tangled Cross-Cutting Concerns",
      "description": "A situation where cross-cutting concerns, such as logging, security, and transactions, are tightly coupled with the business logic.",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "algorithmic-complexity-problems.md",
      "title": "Algorithmic Complexity Problems",
      "description": "Code uses inefficient algorithms or data structures, leading to performance bottlenecks and resource waste.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "legal-disputes.md",
      "title": "Legal Disputes",
      "description": "Conflicts over contracts, deliverables, or responsibilities escalate to legal proceedings, consuming resources and damaging relationships.",
      "category": "Dependencies",
      "size": 17
    },
    {
      "id": "team-coordination-issues.md",
      "title": "Team Coordination Issues",
      "description": "A situation where multiple developers or teams have difficulty working together on the same codebase.",
      "category": "Process",
      "size": 19
    },
    {
      "id": "time-pressure.md",
      "title": "Time Pressure",
      "description": "Teams are forced to take shortcuts to meet immediate deadlines, deferring proper solutions and rushing important tasks like code reviews.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "microservice-communication-overhead.md",
      "title": "Microservice Communication Overhead",
      "description": "Excessive network communication between microservices creates latency, reduces reliability, and impacts overall system performance.",
      "category": "Architecture",
      "size": 15
    },
    {
      "id": "customer-dissatisfaction.md",
      "title": "Customer Dissatisfaction",
      "description": "Users become frustrated with system reliability, performance, or usability issues, leading to complaints and potential customer loss.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "buffer-overflow-vulnerabilities.md",
      "title": "Buffer Overflow Vulnerabilities",
      "description": "Programs write data beyond the boundaries of allocated memory buffers, leading to security vulnerabilities and system instability.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "procedural-background.md",
      "title": "Procedural Background",
      "description": "Developers with a background in procedural programming may struggle to adapt to an object-oriented way of thinking, leading to the creation of procedural-style code in an object-oriented language.",
      "category": "Architecture",
      "size": 16
    },
    {
      "id": "long-release-cycles.md",
      "title": "Long Release Cycles",
      "description": "Releases are delayed due to prolonged manual testing phases or last-minute bug discoveries.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "feature-factory.md",
      "title": "Feature Factory",
      "description": "Organization prioritizes shipping features over understanding their business impact and user value",
      "category": "Management",
      "size": 20
    },
    {
      "id": "cv-driven-development.md",
      "title": "CV Driven Development",
      "description": "Choosing technologies or practices primarily to enhance personal resumes rather than solve business problems",
      "category": "Code",
      "size": 18
    },
    {
      "id": "second-system-effect.md",
      "title": "Second-System Effect",
      "description": "Lessons from an old system lead to overcompensation, creating bloated or overly ambitious designs",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "feature-creep-without-refactoring.md",
      "title": "Feature Creep Without Refactoring",
      "description": "The continuous addition of new features to a codebase without taking the time to refactor and improve the design.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "monolithic-functions-and-classes.md",
      "title": "Monolithic Functions and Classes",
      "description": "Individual functions or classes perform too many unrelated responsibilities, making them difficult to understand and modify.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "log-spam.md",
      "title": "Log Spam",
      "description": "The application or database logs are flooded with a large number of similar-looking queries, making it difficult to identify and diagnose other issues.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "spaghetti-code.md",
      "title": "Spaghetti Code",
      "description": "Code with tangled, unstructured logic that is nearly impossible to understand, debug, or modify safely.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "delayed-value-delivery.md",
      "title": "Delayed Value Delivery",
      "description": "Users have to wait for an extended period to receive new features or bug fixes, leading to frustration and a competitive disadvantage.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "unreleased-resources.md",
      "title": "Unreleased Resources",
      "description": "Objects, connections, file handles, or other system resources are allocated but never properly deallocated or closed.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "index-fragmentation.md",
      "title": "Index Fragmentation",
      "description": "Over time, as data is inserted, updated, and deleted, database indexes become disorganized, reducing their efficiency.",
      "category": "Database",
      "size": 15
    },
    {
      "id": "frequent-changes-to-requirements.md",
      "title": "Frequent Changes to Requirements",
      "description": "The requirements for a project or feature are constantly being updated, even after development has started, leading to rework, delays, and frustration.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "work-blocking.md",
      "title": "Work Blocking",
      "description": "Development tasks cannot proceed without pending approvals, creating bottlenecks and delays in the development process.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "team-demoralization.md",
      "title": "Team Demoralization",
      "description": "Team members lose motivation, confidence, and enthusiasm for their work due to persistent problems or organizational issues.",
      "category": "Culture",
      "size": 18
    },
    {
      "id": "feature-gaps.md",
      "title": "Feature Gaps",
      "description": "Important functionality is missing because developers assumed it wasn't needed, creating incomplete solutions that don't meet user needs.",
      "category": "Business",
      "size": 19
    },
    {
      "id": "high-turnover.md",
      "title": "High Turnover",
      "description": "New developers become frustrated and leave the team due to poor onboarding and system complexity.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "gold-plating.md",
      "title": "Gold Plating",
      "description": "Developers add unnecessary features or complexity to a project because they believe it will impress the stakeholders, even if it was not requested.",
      "category": "Process",
      "size": 18
    },
    {
      "id": "memory-leaks.md",
      "title": "Memory Leaks",
      "description": "Applications fail to release memory that is no longer needed, leading to gradual memory consumption and eventual performance degradation or crashes.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "insufficient-code-review.md",
      "title": "Insufficient Code Review",
      "description": "Code review processes fail to catch design flaws, bugs, or quality issues due to inadequate depth, time, or expertise.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "poor-teamwork.md",
      "title": "Poor Teamwork",
      "description": "Team members work in isolation, resist collaboration, or lack mutual respect, reducing overall effectiveness.",
      "category": "Communication",
      "size": 18
    },
    {
      "id": "god-object-anti-pattern.md",
      "title": "God Object Anti-Pattern",
      "description": "Single classes or components handle too many responsibilities, becoming overly complex and difficult to maintain or test.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "difficult-developer-onboarding.md",
      "title": "Difficult Developer Onboarding",
      "description": "New team members take an unusually long time to become productive due to complex systems, poor documentation, and inadequate onboarding processes.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "technology-isolation.md",
      "title": "Technology Isolation",
      "description": "The system becomes increasingly isolated from modern technology stacks, limiting ability to attract talent and leverage new capabilities.",
      "category": "Code",
      "size": 16
    },
    {
      "id": "accumulation-of-workarounds.md",
      "title": "Accumulation of Workarounds",
      "description": "Instead of fixing core issues, developers create elaborate workarounds that add complexity and technical debt to the system.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "tight-coupling-issues.md",
      "title": "Tight Coupling Issues",
      "description": "Components are overly dependent on each other, making changes difficult and reducing system flexibility and maintainability.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "decision-avoidance.md",
      "title": "Decision Avoidance",
      "description": "Important technical decisions are repeatedly deferred, preventing progress and creating bottlenecks in development work.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "technical-architecture-limitations.md",
      "title": "Technical Architecture Limitations",
      "description": "System architecture design creates constraints that limit performance, scalability, maintainability, or development velocity.",
      "category": "Architecture",
      "size": 17
    },
    {
      "id": "reduced-predictability.md",
      "title": "Reduced Predictability",
      "description": "Development timelines, outcomes, and system behavior become difficult to predict accurately, making planning and expectations management challenging.",
      "category": "Management",
      "size": 18
    },
    {
      "id": "poorly-defined-responsibilities.md",
      "title": "Poorly Defined Responsibilities",
      "description": "Modules or classes are not designed with a single, clear responsibility, leading to confusion and tight coupling.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "new-hire-frustration.md",
      "title": "New Hire Frustration",
      "description": "Recently hired developers experience significant frustration due to barriers preventing them from contributing effectively to the team.",
      "category": "Culture",
      "size": 18
    },
    {
      "id": "feature-creep.md",
      "title": "Feature Creep",
      "description": "The scope of a feature or component gradually expands over time, leading to a complex and bloated system that is difficult to maintain.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "unclear-goals-and-priorities.md",
      "title": "Unclear Goals and Priorities",
      "description": "Constantly shifting priorities and lack of clear direction lead to a sense of futility among team members.",
      "category": "Communication",
      "size": 19
    },
    {
      "id": "requirements-ambiguity.md",
      "title": "Requirements Ambiguity",
      "description": "System requirements are unclear, incomplete, or open to multiple interpretations, leading to misaligned implementations and rework.",
      "category": "Communication",
      "size": 19
    },
    {
      "id": "uncontrolled-codebase-growth.md",
      "title": "Uncontrolled Codebase Growth",
      "description": "A situation where a codebase grows in size and complexity without any control or planning.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "large-feature-scope.md",
      "title": "Large Feature Scope",
      "description": "Features are too large to be broken down into smaller, incremental changes, leading to long-lived branches and integration problems.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "modernization-roi-justification-failure.md",
      "title": "Modernization ROI Justification Failure",
      "description": "Unable to build compelling business cases for legacy modernization due to hidden technical debt and unclear benefit quantification",
      "category": "Business",
      "size": 17
    },
    {
      "id": "work-queue-buildup.md",
      "title": "Work Queue Buildup",
      "description": "Tasks accumulate in queues waiting for bottleneck resources or processes, creating delays and reducing overall system throughput.",
      "category": "Performance",
      "size": 20
    },
    {
      "id": "release-instability.md",
      "title": "Release Instability",
      "description": "Production releases are frequently unstable, causing disruptions for users and requiring immediate attention from the development team.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "scaling-inefficiencies.md",
      "title": "Scaling Inefficiencies",
      "description": "A situation where it is difficult or impossible to scale different parts of a system independently.",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "improper-event-listener-management.md",
      "title": "Improper Event Listener Management",
      "description": "Event listeners are added but not removed when associated objects are destroyed, creating memory leaks and preventing garbage collection.",
      "category": "Architecture",
      "size": 16
    },
    {
      "id": "implementation-rework.md",
      "title": "Implementation Rework",
      "description": "Features must be rebuilt when initial understanding proves incorrect, wasting development effort and delaying delivery.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "increased-time-to-market.md",
      "title": "Increased Time to Market",
      "description": "It takes longer to get new features and products to market, potentially resulting in loss of competitive advantage and revenue opportunities.",
      "category": "Business",
      "size": 17
    },
    {
      "id": "insufficient-audit-logging.md",
      "title": "Insufficient Audit Logging",
      "description": "Inadequate logging of security-relevant events makes it difficult to detect breaches, investigate incidents, or maintain compliance.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "memory-barrier-inefficiency.md",
      "title": "Memory Barrier Inefficiency",
      "description": "Excessive or incorrectly placed memory barriers disrupt CPU pipeline optimization and reduce performance in multi-threaded applications.",
      "category": "Code",
      "size": 14
    },
    {
      "id": "knowledge-sharing-breakdown.md",
      "title": "Knowledge Sharing Breakdown",
      "description": "The process of sharing knowledge and expertise among team members is ineffective, leading to information silos and reduced team learning.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "automated-tooling-ineffectiveness.md",
      "title": "Automated Tooling Ineffectiveness",
      "description": "A situation where automated tooling, such as linters and formatters, is not effective because of the inconsistency of the codebase.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "author-frustration.md",
      "title": "Author Frustration",
      "description": "Developers become frustrated with unpredictable, conflicting, or seemingly arbitrary feedback during the code review process.",
      "category": "Culture",
      "size": 19
    },
    {
      "id": "false-sharing.md",
      "title": "False Sharing",
      "description": "Multiple CPU cores access different variables located on the same cache line, causing unnecessary cache coherency traffic and performance degradation.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "inadequate-code-reviews.md",
      "title": "Inadequate Code Reviews",
      "description": "Code reviews are not consistently performed, are rushed, superficial, or fail to identify critical issues, leading to lower code quality and increased risk.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "data-structure-cache-inefficiency.md",
      "title": "Data Structure Cache Inefficiency",
      "description": "Data structures are organized in ways that cause poor cache performance, leading to excessive memory access latency and reduced throughput.",
      "category": "Code",
      "size": 16
    },
    {
      "id": "high-bug-introduction-rate.md",
      "title": "High Bug Introduction Rate",
      "description": "A high rate of new bugs are introduced with every change to the codebase, indicating underlying quality issues.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "global-state-and-side-effects.md",
      "title": "Global State and Side Effects",
      "description": "Excessive use of global variables or functions with hidden side effects makes it difficult to reason about code behavior.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "nitpicking-culture.md",
      "title": "Nitpicking Culture",
      "description": "Code reviews focus excessively on minor, insignificant details while overlooking important design and functionality issues.",
      "category": "Culture",
      "size": 16
    },
    {
      "id": "poor-project-control.md",
      "title": "Poor Project Control",
      "description": "Project progress is not monitored effectively, allowing problems to go unnoticed until recovery becomes difficult or impossible.",
      "category": "Management",
      "size": 18
    },
    {
      "id": "negative-user-feedback.md",
      "title": "Negative User Feedback",
      "description": "Users complain about slow loading times, application freezes, or other issues, indicating dissatisfaction with the application's performance or usability.",
      "category": "Business",
      "size": 18
    },
    {
      "id": "quality-compromises.md",
      "title": "Quality Compromises",
      "description": "Quality standards are deliberately lowered or shortcuts are taken to meet deadlines, budgets, or other constraints, creating long-term problems.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "legacy-configuration-management-chaos.md",
      "title": "Legacy Configuration Management Chaos",
      "description": "Configuration settings are hardcoded, undocumented, or stored in proprietary formats that prevent modern deployment practices",
      "category": "Code",
      "size": 18
    },
    {
      "id": "reduced-team-productivity.md",
      "title": "Reduced Team Productivity",
      "description": "The overall output and effectiveness of the development team decreases due to various systemic issues and inefficiencies.",
      "category": "Performance",
      "size": 18
    },
    {
      "id": "slow-application-performance.md",
      "title": "Slow Application Performance",
      "description": "User-facing features that rely on the API feel sluggish or unresponsive.",
      "category": "Performance",
      "size": 16
    },
    {
      "id": "large-pull-requests.md",
      "title": "Large Pull Requests",
      "description": "Pull requests are too large to review effectively, leading to superficial reviews and missed issues.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "circular-references.md",
      "title": "Circular References",
      "description": "Two or more objects reference each other in a way that prevents garbage collection, leading to memory leaks and resource exhaustion.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "reviewer-inexperience.md",
      "title": "Reviewer Inexperience",
      "description": "Reviewers lack the experience to identify deeper issues, so they focus on what they understand.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "cross-site-scripting-vulnerabilities.md",
      "title": "Cross-Site Scripting Vulnerabilities",
      "description": "Inadequate input validation and output encoding allows attackers to inject malicious scripts that execute in users' browsers.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "system-integration-blindness.md",
      "title": "System Integration Blindness",
      "description": "Components work correctly in isolation but fail when integrated, revealing gaps in end-to-end system understanding.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "growing-task-queues.md",
      "title": "Growing Task Queues",
      "description": "Asynchronous processing queues accumulate unprocessed tasks, indicating a bottleneck in the processing pipeline.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "regulatory-compliance-drift.md",
      "title": "Regulatory Compliance Drift",
      "description": "Legacy systems fall behind evolving regulatory requirements, creating compliance gaps that are expensive and risky to address",
      "category": "Management",
      "size": 18
    },
    {
      "id": "immature-delivery-strategy.md",
      "title": "Immature Delivery Strategy",
      "description": "Software rollout processes are improvised, inconsistent, or inadequately planned, increasing downtime and user confusion.",
      "category": "Operations",
      "size": 19
    },
    {
      "id": "clever-code.md",
      "title": "Clever Code",
      "description": "Code written to demonstrate technical prowess rather than clarity, making it difficult for others to understand and maintain.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "staff-availability-issues.md",
      "title": "Staff Availability Issues",
      "description": "Needed roles remain unfilled or employees are overbooked, reducing execution capacity and project progress.",
      "category": "Performance",
      "size": 17
    },
    {
      "id": "high-client-side-resource-consumption.md",
      "title": "High Client-Side Resource Consumption",
      "description": "Client applications consume excessive CPU or memory, leading to sluggish performance and poor user experience.",
      "category": "Performance",
      "size": 17
    },
    {
      "id": "long-lived-feature-branches.md",
      "title": "Long-Lived Feature Branches",
      "description": "Code is not being reviewed and merged in a timely manner, leading to integration problems and increased risk.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "poor-communication.md",
      "title": "Poor Communication",
      "description": "Collaboration breaks down as developers become isolated and less willing to engage with peers.",
      "category": "Communication",
      "size": 20
    },
    {
      "id": "hidden-side-effects.md",
      "title": "Hidden Side Effects",
      "description": "Functions have undocumented side effects that modify state or trigger actions beyond their apparent purpose.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "delayed-bug-fixes.md",
      "title": "Delayed Bug Fixes",
      "description": "Known issues remain unresolved for extended periods, causing ongoing problems and user frustration.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "inconsistent-execution.md",
      "title": "Inconsistent Execution",
      "description": "Manual processes lead to variations in how tasks are completed across team members and over time, creating unpredictable outcomes.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "shared-database.md",
      "title": "Shared Database",
      "description": "A situation where multiple services or components share a single database.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "inadequate-initial-reviews.md",
      "title": "Inadequate Initial Reviews",
      "description": "First-round code reviews are incomplete or superficial, failing to identify important issues that are discovered in later review rounds.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "high-defect-rate-in-production.md",
      "title": "High Defect Rate in Production",
      "description": "A large number of bugs are discovered in the live environment after a release, indicating underlying issues in the development and quality assurance process.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "quality-degradation.md",
      "title": "Quality Degradation",
      "description": "System quality decreases over time due to accumulating technical debt, shortcuts, and insufficient quality practices.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "developer-frustration-and-burnout.md",
      "title": "Developer Frustration and Burnout",
      "description": "Developers are demotivated, disengaged, and exhausted due to persistent issues in the work environment and codebase.",
      "category": "Culture",
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
      "id": "cognitive-overload.md",
      "title": "Cognitive Overload",
      "description": "Developers must maintain too many complex systems or concepts in their working memory simultaneously, reducing their effectiveness.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "slow-feature-development.md",
      "title": "Slow Feature Development",
      "description": "The pace of developing and delivering new features is consistently slow, often due to the complexity and fragility of the existing codebase.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "dependency-on-supplier.md",
      "title": "Dependency on Supplier",
      "description": "External vendors control critical parts of the system, reducing organizational flexibility and increasing lock-in risk.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "interrupt-overhead.md",
      "title": "Interrupt Overhead",
      "description": "Excessive hardware interrupts disrupt CPU execution flow, causing frequent context switches and reducing application performance.",
      "category": "Code",
      "size": 12
    },
    {
      "id": "team-dysfunction.md",
      "title": "Team Dysfunction",
      "description": "Team members cannot collaborate effectively due to conflicting goals, interpersonal issues, or structural problems that prevent coordinated effort.",
      "category": "Management",
      "size": 18
    },
    {
      "id": "high-maintenance-costs.md",
      "title": "High Maintenance Costs",
      "description": "A disproportionately large amount of the development budget and effort is consumed by maintaining the existing system rather than creating new value.",
      "category": "Business",
      "size": 17
    },
    {
      "id": "rate-limiting-issues.md",
      "title": "Rate Limiting Issues",
      "description": "Rate limiting mechanisms are misconfigured, too restrictive, or ineffective, causing legitimate requests to be blocked or failing to prevent abuse.",
      "category": "Architecture",
      "size": 16
    },
    {
      "id": "declining-business-metrics.md",
      "title": "Declining Business Metrics",
      "description": "Key business indicators such as user engagement, conversion rates, revenue, and stakeholder confidence deteriorate due to technical problems.",
      "category": "Business",
      "size": 18
    },
    {
      "id": "poor-caching-strategy.md",
      "title": "Poor Caching Strategy",
      "description": "Data that could be cached is fetched from the source on every request, adding unnecessary overhead and increasing latency.",
      "category": "Performance",
      "size": 18
    },
    {
      "id": "system-stagnation.md",
      "title": "System Stagnation",
      "description": "Software systems remain unchanged and fail to evolve to meet new requirements, technologies, or business needs over extended periods.",
      "category": "Business",
      "size": 20
    },
    {
      "id": "insufficient-design-skills.md",
      "title": "Insufficient Design Skills",
      "description": "The development team lacks the necessary skills and experience to design and build well-structured, maintainable software.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "budget-overruns.md",
      "title": "Budget Overruns",
      "description": "The project costs more than originally planned due to the extra work being done, which can lead to a loss of funding and a great deal of frustration for the stakeholders.",
      "category": "Business",
      "size": 16
    },
    {
      "id": "incorrect-max-connection-pool-size.md",
      "title": "Incorrect Max Connection Pool Size",
      "description": "The maximum number of connections in a database connection pool is set incorrectly, leading to either wasted resources or connection exhaustion.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "stack-overflow-errors.md",
      "title": "Stack Overflow Errors",
      "description": "Programs exceed the allocated stack space due to excessive recursion or large local variables, causing application crashes.",
      "category": "Code",
      "size": 14
    },
    {
      "id": "deadlock-conditions.md",
      "title": "Deadlock Conditions",
      "description": "Multiple threads or processes wait indefinitely for each other to release resources, causing system freeze and application unresponsiveness.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "legacy-code-without-tests.md",
      "title": "Legacy Code Without Tests",
      "description": "Existing legacy systems often lack automated tests, making it challenging to add them incrementally and safely modify the code.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "information-fragmentation.md",
      "title": "Information Fragmentation",
      "description": "Critical system knowledge is scattered across multiple locations and formats, making it difficult to find and maintain.",
      "category": "Communication",
      "size": 17
    },
    {
      "id": "rest-api-design-issues.md",
      "title": "REST API Design Issues",
      "description": "Poor REST API design violates REST principles, creates usability problems, and leads to inefficient client-server interactions.",
      "category": "Architecture",
      "size": 17
    },
    {
      "id": "large-estimates-for-small-changes.md",
      "title": "Large Estimates for Small Changes",
      "description": "The team consistently provides large time estimates for seemingly small changes, indicating underlying code complexity and risk.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "project-resource-constraints.md",
      "title": "Project Resource Constraints",
      "description": "Projects cannot obtain necessary budget, personnel, or organizational resources due to poor planning or competing priorities.",
      "category": "Management",
      "size": 18
    },
    {
      "id": "difficult-code-comprehension.md",
      "title": "Difficult Code Comprehension",
      "description": "A situation where developers have a hard time understanding the codebase.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "cache-invalidation-problems.md",
      "title": "Cache Invalidation Problems",
      "description": "Cached data becomes stale or inconsistent with the underlying data source, leading to incorrect application behavior and user confusion.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "fear-of-conflict.md",
      "title": "Fear of Conflict",
      "description": "Reviewers avoid challenging complex logic or design decisions, opting for easier, less confrontational feedback.",
      "category": "Communication",
      "size": 18
    },
    {
      "id": "suboptimal-solutions.md",
      "title": "Suboptimal Solutions",
      "description": "Delivered solutions work but are inefficient, difficult to use, or don't fully address the underlying problems they were meant to solve.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "stakeholder-developer-communication-gap.md",
      "title": "Stakeholder-Developer Communication Gap",
      "description": "A persistent misunderstanding between what stakeholders expect and what the development team builds, leading to rework and dissatisfaction.",
      "category": "Communication",
      "size": 19
    },
    {
      "id": "inadequate-mentoring-structure.md",
      "title": "Inadequate Mentoring Structure",
      "description": "The organization lacks a systematic approach to mentoring new developers, leading to inconsistent guidance and support.",
      "category": "Communication",
      "size": 18
    },
    {
      "id": "poor-test-coverage.md",
      "title": "Poor Test Coverage",
      "description": "Critical parts of the codebase are not covered by tests, creating blind spots in quality assurance.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "team-confusion.md",
      "title": "Team Confusion",
      "description": "Team members are unclear about project goals, priorities, responsibilities, or processes, leading to misaligned efforts and wasted work.",
      "category": "Communication",
      "size": 19
    },
    {
      "id": "market-pressure.md",
      "title": "Market Pressure",
      "description": "External competitive forces or market conditions drive rushed decisions, scope changes, and unrealistic expectations.",
      "category": "Business",
      "size": 17
    },
    {
      "id": "individual-recognition-culture.md",
      "title": "Individual Recognition Culture",
      "description": "A culture where individual accomplishments are valued more than team accomplishments.",
      "category": "Process",
      "size": 15
    },
    {
      "id": "maintenance-bottlenecks.md",
      "title": "Maintenance Bottlenecks",
      "description": "A situation where a small number of developers are the only ones who can make changes to a critical part of the system.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "negative-brand-perception.md",
      "title": "Negative Brand Perception",
      "description": "Users associate the brand with poor quality or unreliability due to technical problems.",
      "category": "Business",
      "size": 16
    },
    {
      "id": "blame-culture.md",
      "title": "Blame Culture",
      "description": "Mistakes are punished instead of addressed constructively, discouraging risk-taking and learning",
      "category": "Management",
      "size": 17
    },
    {
      "id": "overworked-teams.md",
      "title": "Overworked Teams",
      "description": "High workloads lead to burnout, mistakes, and long-term drops in quality and productivity.",
      "category": "Culture",
      "size": 20
    },
    {
      "id": "stagnant-architecture.md",
      "title": "Stagnant Architecture",
      "description": "The system architecture doesn't evolve to meet changing business needs because major refactoring is consistently avoided.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "cascade-delays.md",
      "title": "Cascade Delays",
      "description": "Missed deadlines in one area cause delays in dependent work streams, creating a ripple effect that affects multiple projects and teams.",
      "category": "Business",
      "size": 18
    },
    {
      "id": "technology-stack-fragmentation.md",
      "title": "Technology Stack Fragmentation",
      "description": "Legacy systems create isolated technology islands that prevent standardization and increase operational complexity across the organization",
      "category": "Code",
      "size": 19
    },
    {
      "id": "error-message-information-disclosure.md",
      "title": "Error Message Information Disclosure",
      "description": "Error messages reveal sensitive system information that can be exploited by attackers to understand system architecture and vulnerabilities.",
      "category": "Code",
      "size": 16
    },
    {
      "id": "unclear-documentation-ownership.md",
      "title": "Unclear Documentation Ownership",
      "description": "No clear responsibility for maintaining documentation leads to outdated, inconsistent, or missing information.",
      "category": "Communication",
      "size": 18
    },
    {
      "id": "increased-error-rates.md",
      "title": "Increased Error Rates",
      "description": "An unusual or sustained rise in the frequency of errors reported by an application or service.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "network-latency.md",
      "title": "Network Latency",
      "description": "Delays in data transmission across the network significantly increase response times and impact application performance.",
      "category": "Performance",
      "size": 16
    },
    {
      "id": "inefficient-processes.md",
      "title": "Inefficient Processes",
      "description": "Poor workflows, excessive meetings, or bureaucratic procedures waste development time and reduce team productivity.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "reduced-innovation.md",
      "title": "Reduced Innovation",
      "description": "Teams become resistant to new ideas and focus only on the bare minimum to get by.",
      "category": "Business",
      "size": 16
    },
    {
      "id": "fear-of-change.md",
      "title": "Fear of Change",
      "description": "Developers are hesitant to modify existing code due to the high risk of breaking something.",
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
      "id": "increased-cost-of-development.md",
      "title": "Increased Cost of Development",
      "description": "The cost of fixing bugs and maintaining poor code is significantly higher than preventing issues initially.",
      "category": "Business",
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
      "id": "delayed-project-timelines.md",
      "title": "Delayed Project Timelines",
      "description": "Projects consistently take longer than planned, missing deadlines and extending delivery schedules beyond original estimates.",
      "category": "Process",
      "size": 20
    },
    {
      "id": "schema-evolution-paralysis.md",
      "title": "Schema Evolution Paralysis",
      "description": "Database schema cannot be modified to support new requirements due to extensive dependencies and lack of migration tooling",
      "category": "Code",
      "size": 19
    },
    {
      "id": "rapid-system-changes.md",
      "title": "Rapid System Changes",
      "description": "Frequent modifications to system architecture, APIs, or core functionality outpace documentation and team understanding.",
      "category": "Communication",
      "size": 18
    },
    {
      "id": "product-direction-chaos.md",
      "title": "Product Direction Chaos",
      "description": "Multiple stakeholders provide conflicting priorities without clear product leadership, causing team confusion and wasted effort.",
      "category": "Business",
      "size": 18
    },
    {
      "id": "decision-paralysis.md",
      "title": "Decision Paralysis",
      "description": "Developers struggle to make choices without clear guidance, which can lead to a slowdown in development and a great deal of frustration for the development team.",
      "category": "Management",
      "size": 19
    },
    {
      "id": "mentor-burnout.md",
      "title": "Mentor Burnout",
      "description": "Experienced team members become overwhelmed and exhausted from excessive mentoring and knowledge transfer responsibilities.",
      "category": "Culture",
      "size": 17
    },
    {
      "id": "fear-of-breaking-changes.md",
      "title": "Fear of Breaking Changes",
      "description": "The team is reluctant to make changes to the codebase for fear of breaking existing functionality, which can lead to a stagnant and outdated system.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "external-service-delays.md",
      "title": "External Service Delays",
      "description": "An API depends on other services (third-party or internal) that are slow to respond, causing the API itself to be slow.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "unrealistic-schedule.md",
      "title": "Unrealistic Schedule",
      "description": "Project timelines are based on optimistic assumptions rather than realistic estimates, leading to stress and shortcuts.",
      "category": "Management",
      "size": 19
    },
    {
      "id": "sql-injection-vulnerabilities.md",
      "title": "SQL Injection Vulnerabilities",
      "description": "Inadequate input sanitization allows attackers to inject malicious SQL code, potentially compromising database security and data integrity.",
      "category": "Database",
      "size": 14
    },
    {
      "id": "integer-overflow-underflow.md",
      "title": "Integer Overflow and Underflow",
      "description": "Arithmetic operations produce results that exceed the maximum or minimum values representable by integer data types, leading to unexpected behavior.",
      "category": "Code",
      "size": 16
    },
    {
      "id": "environment-variable-issues.md",
      "title": "Environment Variable Issues",
      "description": "Improper management of environment variables causes configuration problems, security vulnerabilities, and deployment failures.",
      "category": "Operations",
      "size": 19
    },
    {
      "id": "inconsistent-onboarding-experience.md",
      "title": "Inconsistent Onboarding Experience",
      "description": "New team members receive dramatically different onboarding experiences depending on who is available to help them, creating unequal outcomes.",
      "category": "Communication",
      "size": 18
    },
    {
      "id": "misunderstanding-of-oop.md",
      "title": "Misunderstanding of OOP",
      "description": "A lack of understanding of the fundamental principles of object-oriented programming can lead to the creation of poorly designed and difficult-to-maintain code.",
      "category": "Architecture",
      "size": 20
    },
    {
      "id": "authentication-bypass-vulnerabilities.md",
      "title": "Authentication Bypass Vulnerabilities",
      "description": "Security flaws that allow attackers to bypass authentication mechanisms and gain unauthorized access to protected resources.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "serialization-deserialization-bottlenecks.md",
      "title": "Serialization/Deserialization Bottlenecks",
      "description": "Inefficient serialization and deserialization of data creates performance bottlenecks in API communications and data persistence operations.",
      "category": "Architecture",
      "size": 17
    },
    {
      "id": "missing-end-to-end-tests.md",
      "title": "Missing End-to-End Tests",
      "description": "Complete user flows are not tested from start to finish, allowing workflow-breaking bugs to reach production.",
      "category": "Code",
      "size": 16
    },
    {
      "id": "frequent-hotfixes-and-rollbacks.md",
      "title": "Frequent Hotfixes and Rollbacks",
      "description": "The team is constantly deploying small fixes or rolling back releases due to insufficient testing and quality control.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "memory-swapping.md",
      "title": "Memory Swapping",
      "description": "The database server runs out of physical memory and starts using disk swap space, which dramatically slows down performance.",
      "category": "Performance",
      "size": 17
    },
    {
      "id": "task-queues-backing-up.md",
      "title": "Task Queues Backing Up",
      "description": "Asynchronous jobs or messages take longer to process, causing queues to grow and delaying critical operations.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "silent-data-corruption.md",
      "title": "Silent Data Corruption",
      "description": "Data becomes corrupted without triggering errors or alerts, leading to incorrect results and loss of data integrity.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "unproductive-meetings.md",
      "title": "Unproductive Meetings",
      "description": "Meetings that consume significant time but yield little progress or concrete decisions.",
      "category": "Communication",
      "size": 18
    },
    {
      "id": "regression-bugs.md",
      "title": "Regression Bugs",
      "description": "New features or fixes inadvertently break existing functionality that was previously working correctly.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "database-query-performance-issues.md",
      "title": "Database Query Performance Issues",
      "description": "Poorly optimized database queries cause slow response times, high resource consumption, and scalability problems.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "skill-development-gaps.md",
      "title": "Skill Development Gaps",
      "description": "Team members don't develop expertise in important technologies or domains due to avoidance, specialization, or inadequate learning opportunities.",
      "category": "Team",
      "size": 16
    },
    {
      "id": "scope-creep.md",
      "title": "Scope Creep",
      "description": "Project requirements expand continuously without proper control or impact analysis, threatening timelines, budgets, and the original objectives.",
      "category": "Process",
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
      "id": "reduced-feature-quality.md",
      "title": "Reduced Feature Quality",
      "description": "Less time is available for polish and refinement of delivered features, resulting in lower-quality user experiences and functionality.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "flaky-tests.md",
      "title": "Flaky Tests",
      "description": "Tests fail randomly due to timing, setup, or dependencies, undermining trust in the test suite",
      "category": "Code",
      "size": 19
    },
    {
      "id": "process-design-flaws.md",
      "title": "Process Design Flaws",
      "description": "Development processes are poorly designed, creating inefficiencies, bottlenecks, and obstacles to productive work.",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "implementation-starts-without-design.md",
      "title": "Implementation Starts Without Design",
      "description": "Development begins with unclear structure, leading to disorganized code and architectural drift",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "inadequate-requirements-gathering.md",
      "title": "Inadequate Requirements Gathering",
      "description": "Insufficient analysis and documentation of requirements leads to building solutions that don't meet actual needs.",
      "category": "Process",
      "size": 19
    },
    {
      "id": "inefficient-frontend-code.md",
      "title": "Inefficient Frontend Code",
      "description": "Unoptimized JavaScript, excessive DOM manipulation, or complex CSS animations that are computationally expensive.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "stakeholder-confidence-loss.md",
      "title": "Stakeholder Confidence Loss",
      "description": "Business partners lose trust in the development team's ability to deliver on commitments, creating tension and reduced collaboration.",
      "category": "Business",
      "size": 19
    },
    {
      "id": "queries-that-prevent-index-usage.md",
      "title": "Queries That Prevent Index Usage",
      "description": "The way a query is written can prevent the database from using an available index, forcing slower full-table scans or less efficient index scans.",
      "category": "Performance",
      "size": 16
    },
    {
      "id": "inadequate-configuration-management.md",
      "title": "Inadequate Configuration Management",
      "description": "Versions of code, data, or infrastructure are not tracked properly, leading to errors or rollback issues",
      "category": "Code",
      "size": 18
    },
    {
      "id": "data-migration-integrity-issues.md",
      "title": "Data Migration Integrity Issues",
      "description": "Data loses integrity, consistency, or meaning during migration from legacy to modern systems due to schema mismatches and format incompatibilities",
      "category": "Code",
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
      "id": "no-continuous-feedback-loop.md",
      "title": "No Continuous Feedback Loop",
      "description": "Stakeholders are not involved throughout the development process, and feedback is only gathered at the very end, leading to misaligned deliverables.",
      "category": "Communication",
      "size": 17
    },
    {
      "id": "database-connection-leaks.md",
      "title": "Database Connection Leaks",
      "description": "Database connections are opened but not properly closed, leading to connection pool exhaustion and application failures.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "increased-customer-support-load.md",
      "title": "Increased Customer Support Load",
      "description": "Users contact support more frequently due to frustration or inability to complete tasks.",
      "category": "Business",
      "size": 15
    },
    {
      "id": "service-timeouts.md",
      "title": "Service Timeouts",
      "description": "Services fail to complete requests within an acceptable time limit, causing errors, cascading failures, and system instability.",
      "category": "Code",
      "size": 18
    },
    {
      "id": "testing-environment-fragility.md",
      "title": "Testing Environment Fragility",
      "description": "Testing infrastructure is unreliable, difficult to maintain, and fails to accurately represent production conditions, undermining test effectiveness.",
      "category": "Operations",
      "size": 19
    },
    {
      "id": "virtual-memory-thrashing.md",
      "title": "Virtual Memory Thrashing",
      "description": "System constantly swaps pages between physical memory and disk, causing severe performance degradation due to excessive paging activity.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "large-risky-releases.md",
      "title": "Large, Risky Releases",
      "description": "Infrequent releases lead to large, complex deployments that are difficult to test, prone to failure, and have a significant impact on users when they go wrong.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "uneven-workload-distribution.md",
      "title": "Uneven Workload Distribution",
      "description": "Work is not distributed fairly or effectively across team members, leading to some being overloaded while others are underutilized.",
      "category": "Performance",
      "size": 17
    },
    {
      "id": "race-conditions.md",
      "title": "Race Conditions",
      "description": "Multiple threads access shared resources simultaneously without proper synchronization, leading to unpredictable behavior and data corruption.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "legacy-business-logic-extraction-difficulty.md",
      "title": "Legacy Business Logic Extraction Difficulty",
      "description": "Critical business rules are embedded deep within legacy code structures, making them nearly impossible to identify and extract",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "alignment-and-padding-issues.md",
      "title": "Alignment and Padding Issues",
      "description": "Data structures have inefficient memory layout due to poor alignment and excessive padding, wasting memory and reducing cache efficiency.",
      "category": "Architecture",
      "size": 15
    },
    {
      "id": "complex-deployment-process.md",
      "title": "Complex Deployment Process",
      "description": "The process of deploying software to production is manual, time-consuming, and error-prone, which contributes to long release cycles and a high risk of failure.",
      "category": "Operations",
      "size": 20
    },
    {
      "id": "api-versioning-conflicts.md",
      "title": "API Versioning Conflicts",
      "description": "Inconsistent or poorly managed API versioning creates compatibility issues, breaking changes, and integration failures between services.",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "lack-of-ownership-and-accountability.md",
      "title": "Lack of Ownership and Accountability",
      "description": "No clear responsibility for maintaining code quality, documentation, or specific system components over time.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "inconsistent-naming-conventions.md",
      "title": "Inconsistent Naming Conventions",
      "description": "Unstructured or conflicting names make code harder to read, navigate, and maintain",
      "category": "Code",
      "size": 17
    },
    {
      "id": "incomplete-knowledge.md",
      "title": "Incomplete Knowledge",
      "description": "Developers are unaware of all the locations where similar logic exists, which can lead to synchronization problems and other issues.",
      "category": "Communication",
      "size": 18
    },
    {
      "id": "inefficient-database-indexing.md",
      "title": "Inefficient Database Indexing",
      "description": "The database lacks appropriate indexes for common query patterns, forcing slow, full-table scans for data retrieval operations.",
      "category": "Database",
      "size": 18
    },
    {
      "id": "inconsistent-quality.md",
      "title": "Inconsistent Quality",
      "description": "Some parts of the system are well-maintained while others deteriorate, creating unpredictable user experiences and maintenance challenges.",
      "category": "Code",
      "size": 20
    },
    {
      "id": "difficulty-quantifying-benefits.md",
      "title": "Difficulty Quantifying Benefits",
      "description": "It is hard to measure the ROI of refactoring work compared to new features, so technical improvements often lose out in prioritization decisions.",
      "category": "Business",
      "size": 19
    },
    {
      "id": "excessive-object-allocation.md",
      "title": "Excessive Object Allocation",
      "description": "Code creates a large number of temporary objects, putting pressure on the garbage collector and degrading performance.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "imperative-data-fetching-logic.md",
      "title": "Imperative Data Fetching Logic",
      "description": "The application code is written in a way that fetches data in a loop, rather than using a more efficient, declarative approach, leading to performance problems.",
      "category": "Architecture",
      "size": 17
    },
    {
      "id": "memory-fragmentation.md",
      "title": "Memory Fragmentation",
      "description": "Available memory becomes divided into small, non-contiguous blocks, preventing allocation of larger objects despite sufficient total free memory.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "technology-lock-in.md",
      "title": "Technology Lock-In",
      "description": "A situation where it is difficult or impossible to switch to a new technology because of the high cost or effort involved.",
      "category": "Architecture",
      "size": 17
    },
    {
      "id": "excessive-logging.md",
      "title": "Excessive Logging",
      "description": "Applications generate a very high volume of logs, consuming excessive disk space and potentially impacting performance.",
      "category": "Code",
      "size": 19
    },
    {
      "id": "service-discovery-failures.md",
      "title": "Service Discovery Failures",
      "description": "Service discovery mechanisms fail to locate or connect to services, causing communication failures and system instability in distributed architectures.",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "context-switching-overhead.md",
      "title": "Context Switching Overhead",
      "description": "Developers must constantly switch between different tools, systems, or problem domains, reducing productivity and increasing cognitive load.",
      "category": "Process",
      "size": 19
    },
    {
      "id": "resistance-to-change.md",
      "title": "Resistance to Change",
      "description": "Teams are hesitant to refactor or improve parts of the system due to perceived risk and effort, leading to stagnation.",
      "category": "Code",
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
      "id": "endianness-conversion-overhead.md",
      "title": "Endianness Conversion Overhead",
      "description": "Frequent byte order conversions between different endianness formats create performance overhead in data processing and network communication.",
      "category": "Code",
      "size": 17
    },
    {
      "id": "strangler-fig-pattern-failures.md",
      "title": "Strangler Fig Pattern Failures",
      "description": "Incremental modernization using the strangler fig pattern stalls due to complex interdependencies and data consistency challenges",
      "category": "Architecture",
      "size": 19
    },
    {
      "id": "deployment-risk.md",
      "title": "Deployment Risk",
      "description": "System deployments carry high risk of failure or damage due to irreversible changes and lack of recovery mechanisms.",
      "category": "Management",
      "size": 20
    },
    {
      "id": "unpredictable-system-behavior.md",
      "title": "Unpredictable System Behavior",
      "description": "Changes in one part of the system have unexpected side effects in seemingly unrelated areas due to hidden dependencies.",
      "category": "Architecture",
      "size": 18
    },
    {
      "id": "slow-response-times-for-lists.md",
      "title": "Slow Response Times for Lists",
      "description": "Web pages or API endpoints that display lists of items are significantly slower to load than those that display single items, often due to inefficient data fetching.",
      "category": "Database",
      "size": 16
    },
    {
      "id": "unmotivated-employees.md",
      "title": "Unmotivated Employees",
      "description": "Team members lack engagement and enthusiasm, contributing minimally to project success and affecting overall team morale.",
      "category": "Culture",
      "size": 17
    },
    {
      "id": "knowledge-silos.md",
      "title": "Knowledge Silos",
      "description": "Important research findings and expertise remain isolated to individual team members, preventing knowledge sharing and team learning.",
      "category": "Culture",
      "size": 20
    },
    {
      "id": "team-silos.md",
      "title": "Team Silos",
      "description": "Development teams or individual developers work in isolation, leading to duplicated effort, inconsistent solutions, and a lack of knowledge sharing.",
      "category": "Communication",
      "size": 18
    },
    {
      "id": "poor-contract-design.md",
      "title": "Poor Contract Design",
      "description": "Legal agreements and contracts don't reflect project realities, technical requirements, or allow for necessary flexibility during development.",
      "category": "Management",
      "size": 18
    },
    {
      "id": "authorization-flaws.md",
      "title": "Authorization Flaws",
      "description": "Inadequate access control mechanisms allow users to perform actions or access resources beyond their intended permissions.",
      "category": "Code",
      "size": 18
    }
  ],
  "links": [
    {
      "source": "accumulated-decision-debt.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "accumulated-decision-debt.md",
      "target": "decision-avoidance.md"
    },
    {
      "source": "accumulated-decision-debt.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "accumulated-decision-debt.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "accumulated-decision-debt.md",
      "target": "dependency-on-supplier.md"
    },
    {
      "source": "accumulated-decision-debt.md",
      "target": "legacy-api-versioning-nightmare.md"
    },
    {
      "source": "accumulated-decision-debt.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "accumulated-decision-debt.md",
      "target": "technical-architecture-limitations.md"
    },
    {
      "source": "accumulated-decision-debt.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "complex-implementation-paths.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "delayed-bug-fixes.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "hidden-dependencies.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "implementation-starts-without-design.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "inappropriate-skillset.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "integration-difficulties.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "legacy-api-versioning-nightmare.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "market-pressure.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "poor-domain-model.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "procrastination-on-complex-tasks.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "reduced-feature-quality.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "regulatory-compliance-drift.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "schema-evolution-paralysis.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "scope-change-resistance.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "stagnant-architecture.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "team-churn-impact.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "team-dysfunction.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "technical-architecture-limitations.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "accumulation-of-workarounds.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "algorithmic-complexity-problems.md",
      "target": "data-structure-cache-inefficiency.md"
    },
    {
      "source": "algorithmic-complexity-problems.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "algorithmic-complexity-problems.md",
      "target": "excessive-object-allocation.md"
    },
    {
      "source": "algorithmic-complexity-problems.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "algorithmic-complexity-problems.md",
      "target": "inefficient-code.md"
    },
    {
      "source": "algorithmic-complexity-problems.md",
      "target": "serialization-deserialization-bottlenecks.md"
    },
    {
      "source": "algorithmic-complexity-problems.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "algorithmic-complexity-problems.md",
      "target": "unbounded-data-structures.md"
    },
    {
      "source": "alignment-and-padding-issues.md",
      "target": "data-structure-cache-inefficiency.md"
    },
    {
      "source": "alignment-and-padding-issues.md",
      "target": "dma-coherency-issues.md"
    },
    {
      "source": "alignment-and-padding-issues.md",
      "target": "false-sharing.md"
    },
    {
      "source": "alignment-and-padding-issues.md",
      "target": "memory-fragmentation.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "decision-avoidance.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "fear-of-failure.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "analysis-paralysis.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "api-versioning-conflicts.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "api-versioning-conflicts.md",
      "target": "rest-api-design-issues.md"
    },
    {
      "source": "approval-dependencies.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "approval-dependencies.md",
      "target": "cascade-delays.md"
    },
    {
      "source": "approval-dependencies.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "approval-dependencies.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "approval-dependencies.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "approval-dependencies.md",
      "target": "micromanagement-culture.md"
    },
    {
      "source": "approval-dependencies.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "approval-dependencies.md",
      "target": "scope-change-resistance.md"
    },
    {
      "source": "approval-dependencies.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "approval-dependencies.md",
      "target": "work-blocking.md"
    },
    {
      "source": "architectural-mismatch.md",
      "target": "complex-implementation-paths.md"
    },
    {
      "source": "architectural-mismatch.md",
      "target": "integration-difficulties.md"
    },
    {
      "source": "architectural-mismatch.md",
      "target": "microservice-communication-overhead.md"
    },
    {
      "source": "architectural-mismatch.md",
      "target": "schema-evolution-paralysis.md"
    },
    {
      "source": "architectural-mismatch.md",
      "target": "stagnant-architecture.md"
    },
    {
      "source": "architectural-mismatch.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "feature-gaps.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "implementation-rework.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "inadequate-requirements-gathering.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "poor-communication.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "second-system-effect.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "stakeholder-developer-communication-gap.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "assumption-based-development.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "authentication-bypass-vulnerabilities.md",
      "target": "error-message-information-disclosure.md"
    },
    {
      "source": "authentication-bypass-vulnerabilities.md",
      "target": "insecure-data-transmission.md"
    },
    {
      "source": "authentication-bypass-vulnerabilities.md",
      "target": "password-security-weaknesses.md"
    },
    {
      "source": "authentication-bypass-vulnerabilities.md",
      "target": "secret-management-problems.md"
    },
    {
      "source": "authentication-bypass-vulnerabilities.md",
      "target": "session-management-issues.md"
    },
    {
      "source": "author-frustration.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "author-frustration.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "author-frustration.md",
      "target": "extended-review-cycles.md"
    },
    {
      "source": "author-frustration.md",
      "target": "inadequate-initial-reviews.md"
    },
    {
      "source": "author-frustration.md",
      "target": "perfectionist-review-culture.md"
    },
    {
      "source": "author-frustration.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "author-frustration.md",
      "target": "review-process-avoidance.md"
    },
    {
      "source": "author-frustration.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "authorization-flaws.md",
      "target": "data-protection-risk.md"
    },
    {
      "source": "authorization-flaws.md",
      "target": "session-management-issues.md"
    },
    {
      "source": "automated-tooling-ineffectiveness.md",
      "target": "inefficient-development-environment.md"
    },
    {
      "source": "automated-tooling-ineffectiveness.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "automated-tooling-ineffectiveness.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "avoidance-behaviors.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "avoidance-behaviors.md",
      "target": "blame-culture.md"
    },
    {
      "source": "avoidance-behaviors.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "avoidance-behaviors.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "avoidance-behaviors.md",
      "target": "fear-of-failure.md"
    },
    {
      "source": "avoidance-behaviors.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "bikeshedding.md",
      "target": "unproductive-meetings.md"
    },
    {
      "source": "blame-culture.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "blame-culture.md",
      "target": "decision-avoidance.md"
    },
    {
      "source": "blame-culture.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "blame-culture.md",
      "target": "defensive-coding-practices.md"
    },
    {
      "source": "blame-culture.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "blame-culture.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "blame-culture.md",
      "target": "fear-of-conflict.md"
    },
    {
      "source": "blame-culture.md",
      "target": "fear-of-failure.md"
    },
    {
      "source": "blame-culture.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "blame-culture.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "blame-culture.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "blame-culture.md",
      "target": "poor-teamwork.md"
    },
    {
      "source": "blame-culture.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "blame-culture.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "blame-culture.md",
      "target": "reviewer-anxiety.md"
    },
    {
      "source": "bloated-class.md",
      "target": "merge-conflicts.md"
    },
    {
      "source": "bottleneck-formation.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "bottleneck-formation.md",
      "target": "capacity-mismatch.md"
    },
    {
      "source": "bottleneck-formation.md",
      "target": "cascade-delays.md"
    },
    {
      "source": "bottleneck-formation.md",
      "target": "inconsistent-knowledge-acquisition.md"
    },
    {
      "source": "bottleneck-formation.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "bottleneck-formation.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "bottleneck-formation.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "bottleneck-formation.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "bottleneck-formation.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "bottleneck-formation.md",
      "target": "work-blocking.md"
    },
    {
      "source": "bottleneck-formation.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "breaking-changes.md",
      "target": "abi-compatibility-issues.md"
    },
    {
      "source": "breaking-changes.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "breaking-changes.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "breaking-changes.md",
      "target": "legacy-api-versioning-nightmare.md"
    },
    {
      "source": "breaking-changes.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "hardcoded-values.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "regression-bugs.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "single-entry-point-design.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "brittle-codebase.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "cascade-delays.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "changing-project-scope.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "gold-plating.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "inadequate-requirements-gathering.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "maintenance-cost-increase.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "poor-planning.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "poor-project-control.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "resource-waste.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "scope-creep.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "budget-overruns.md",
      "target": "strangler-fig-pattern-failures.md"
    },
    {
      "source": "buffer-overflow-vulnerabilities.md",
      "target": "integer-overflow-underflow.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "insufficient-worker-capacity.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "resource-contention.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "resource-waste.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "capacity-mismatch.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "cargo-culting.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "cargo-culting.md",
      "target": "alignment-and-padding-issues.md"
    },
    {
      "source": "cargo-culting.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "cargo-culting.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "capacity-mismatch.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "poor-project-control.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "cascade-delays.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "abi-compatibility-issues.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "buffer-overflow-vulnerabilities.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "hidden-dependencies.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "high-connection-count.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "inadequate-error-handling.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "inadequate-integration-tests.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "insufficient-worker-capacity.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "microservice-communication-overhead.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "resource-contention.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "service-timeouts.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "system-integration-blindness.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "system-outages.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "unbounded-data-growth.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "unbounded-data-structures.md"
    },
    {
      "source": "cascade-failures.md",
      "target": "upstream-timeouts.md"
    },
    {
      "source": "change-management-chaos.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "change-management-chaos.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "change-management-chaos.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "changing-project-scope.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "changing-project-scope.md",
      "target": "product-direction-chaos.md"
    },
    {
      "source": "changing-project-scope.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "changing-project-scope.md",
      "target": "time-pressure.md"
    },
    {
      "source": "changing-project-scope.md",
      "target": "unclear-goals-and-priorities.md"
    },
    {
      "source": "circular-references.md",
      "target": "garbage-collection-pressure.md"
    },
    {
      "source": "clever-code.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "clever-code.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "code-duplication.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "code-duplication.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "code-duplication.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "code-duplication.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "code-duplication.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "code-duplication.md",
      "target": "duplicated-work.md"
    },
    {
      "source": "code-duplication.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "code-duplication.md",
      "target": "hardcoded-values.md"
    },
    {
      "source": "code-duplication.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "code-duplication.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "code-duplication.md",
      "target": "inconsistent-naming-conventions.md"
    },
    {
      "source": "code-duplication.md",
      "target": "inexperienced-developers.md"
    },
    {
      "source": "code-duplication.md",
      "target": "maintenance-cost-increase.md"
    },
    {
      "source": "code-duplication.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "code-duplication.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "code-duplication.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "code-duplication.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "code-duplication.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "code-duplication.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "code-duplication.md",
      "target": "synchronization-problems.md"
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
      "source": "code-review-inefficiency.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "code-review-inefficiency.md",
      "target": "inadequate-initial-reviews.md"
    },
    {
      "source": "code-review-inefficiency.md",
      "target": "inconsistent-coding-standards.md"
    },
    {
      "source": "code-review-inefficiency.md",
      "target": "nitpicking-culture.md"
    },
    {
      "source": "code-review-inefficiency.md",
      "target": "reduced-code-submission-frequency.md"
    },
    {
      "source": "code-review-inefficiency.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "code-review-inefficiency.md",
      "target": "undefined-code-style-guidelines.md"
    },
    {
      "source": "cognitive-overload.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "cognitive-overload.md",
      "target": "complex-domain-model.md"
    },
    {
      "source": "cognitive-overload.md",
      "target": "context-switching-overhead.md"
    },
    {
      "source": "cognitive-overload.md",
      "target": "mental-fatigue.md"
    },
    {
      "source": "cognitive-overload.md",
      "target": "procrastination-on-complex-tasks.md"
    },
    {
      "source": "communication-breakdown.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "communication-breakdown.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "communication-breakdown.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "communication-breakdown.md",
      "target": "duplicated-research-effort.md"
    },
    {
      "source": "communication-breakdown.md",
      "target": "duplicated-work.md"
    },
    {
      "source": "communication-breakdown.md",
      "target": "language-barriers.md"
    },
    {
      "source": "communication-breakdown.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "communication-breakdown.md",
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "communication-breakdown.md",
      "target": "stakeholder-frustration.md"
    },
    {
      "source": "communication-breakdown.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "communication-risk-outside-project.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "communication-risk-within-project.md",
      "target": "communication-risk-outside-project.md"
    },
    {
      "source": "competing-priorities.md",
      "target": "context-switching-overhead.md"
    },
    {
      "source": "competing-priorities.md",
      "target": "market-pressure.md"
    },
    {
      "source": "competing-priorities.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "competing-priorities.md",
      "target": "power-struggles.md"
    },
    {
      "source": "competing-priorities.md",
      "target": "project-authority-vacuum.md"
    },
    {
      "source": "competing-priorities.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "competing-priorities.md",
      "target": "unclear-goals-and-priorities.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "feature-bloat.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "feature-gaps.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "inadequate-onboarding.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "integration-difficulties.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "long-release-cycles.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "maintenance-cost-increase.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "modernization-roi-justification-failure.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "negative-brand-perception.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "negative-user-feedback.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "obsolete-technologies.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "reduced-innovation.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "regulatory-compliance-drift.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "system-outages.md"
    },
    {
      "source": "competitive-disadvantage.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "complex-and-obscure-logic.md",
      "target": "authorization-flaws.md"
    },
    {
      "source": "complex-and-obscure-logic.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "complex-and-obscure-logic.md",
      "target": "complex-domain-model.md"
    },
    {
      "source": "complex-and-obscure-logic.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "complex-and-obscure-logic.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "complex-and-obscure-logic.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "complex-and-obscure-logic.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "complex-and-obscure-logic.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "complex-and-obscure-logic.md",
      "target": "increased-risk-of-bugs.md"
    },
    {
      "source": "complex-and-obscure-logic.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "complex-and-obscure-logic.md",
      "target": "stack-overflow-errors.md"
    },
    {
      "source": "complex-and-obscure-logic.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "complex-and-obscure-logic.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "complex-and-obscure-logic.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "complex-deployment-process.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "complex-deployment-process.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "complex-deployment-process.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "complex-deployment-process.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "complex-deployment-process.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "complex-deployment-process.md",
      "target": "legacy-configuration-management-chaos.md"
    },
    {
      "source": "complex-deployment-process.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "complex-deployment-process.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "complex-deployment-process.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "cross-system-data-synchronization-problems.md"
    },
    {
      "source": "complex-domain-model.md",
      "target": "strangler-fig-pattern-failures.md"
    },
    {
      "source": "complex-implementation-paths.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "configuration-chaos.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "configuration-chaos.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "configuration-chaos.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "configuration-chaos.md",
      "target": "insecure-data-transmission.md"
    },
    {
      "source": "configuration-chaos.md",
      "target": "logging-configuration-issues.md"
    },
    {
      "source": "configuration-chaos.md",
      "target": "secret-management-problems.md"
    },
    {
      "source": "configuration-drift.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "configuration-drift.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "configuration-drift.md",
      "target": "inadequate-configuration-management.md"
    },
    {
      "source": "configuration-drift.md",
      "target": "legacy-configuration-management-chaos.md"
    },
    {
      "source": "configuration-drift.md",
      "target": "rate-limiting-issues.md"
    },
    {
      "source": "configuration-drift.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "configuration-drift.md",
      "target": "testing-environment-fragility.md"
    },
    {
      "source": "conflicting-reviewer-opinions.md",
      "target": "author-frustration.md"
    },
    {
      "source": "conflicting-reviewer-opinions.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "conflicting-reviewer-opinions.md",
      "target": "extended-review-cycles.md"
    },
    {
      "source": "conflicting-reviewer-opinions.md",
      "target": "inconsistent-naming-conventions.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "context-switching-overhead.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "developer-frustration-and-burnout.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "development-disruption.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "increased-risk-of-bugs.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "poor-test-coverage.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "priority-thrashing.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "quality-compromises.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "reduced-feature-quality.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "reduced-innovation.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "regression-bugs.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "system-outages.md"
    },
    {
      "source": "constant-firefighting.md",
      "target": "time-pressure.md"
    },
    {
      "source": "constantly-shifting-deadlines.md",
      "target": "cascade-delays.md"
    },
    {
      "source": "constantly-shifting-deadlines.md",
      "target": "changing-project-scope.md"
    },
    {
      "source": "constantly-shifting-deadlines.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "constantly-shifting-deadlines.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "constantly-shifting-deadlines.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "constantly-shifting-deadlines.md",
      "target": "stakeholder-frustration.md"
    },
    {
      "source": "constantly-shifting-deadlines.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "competing-priorities.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "extended-review-cycles.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "mental-fatigue.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "no-formal-change-control-process.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "priority-thrashing.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "slow-development-velocity.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "work-blocking.md"
    },
    {
      "source": "context-switching-overhead.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "convenience-driven-development.md",
      "target": "code-duplication.md"
    },
    {
      "source": "convenience-driven-development.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "convenience-driven-development.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "convenience-driven-development.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "convenience-driven-development.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "convenience-driven-development.md",
      "target": "poor-domain-model.md"
    },
    {
      "source": "convenience-driven-development.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "convenience-driven-development.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "convenience-driven-development.md",
      "target": "secret-management-problems.md"
    },
    {
      "source": "convenience-driven-development.md",
      "target": "shared-database.md"
    },
    {
      "source": "convenience-driven-development.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "copy-paste-programming.md",
      "target": "code-duplication.md"
    },
    {
      "source": "copy-paste-programming.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "cross-system-data-synchronization-problems.md",
      "target": "strangler-fig-pattern-failures.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "cross-site-scripting-vulnerabilities.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "cross-system-data-synchronization-problems.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "data-migration-integrity-issues.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "feature-bloat.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "feature-factory.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "feature-gaps.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "high-defect-rate-in-production.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "high-resource-utilization-on-client.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "inability-to-innovate.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "inadequate-error-handling.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "inadequate-onboarding.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "increased-error-rates.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "insecure-data-transmission.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "long-release-cycles.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "negative-brand-perception.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "password-security-weaknesses.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "reduced-feature-quality.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "regression-bugs.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "service-timeouts.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "system-outages.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "user-confusion.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "user-frustration.md"
    },
    {
      "source": "customer-dissatisfaction.md",
      "target": "user-trust-erosion.md"
    },
    {
      "source": "cv-driven-development.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "cv-driven-development.md",
      "target": "clever-code.md"
    },
    {
      "source": "data-migration-complexities.md",
      "target": "data-migration-integrity-issues.md"
    },
    {
      "source": "data-migration-complexities.md",
      "target": "database-schema-design-problems.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "cross-system-data-synchronization-problems.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "data-migration-integrity-issues.md",
      "target": "database-schema-design-problems.md"
    },
    {
      "source": "data-protection-risk.md",
      "target": "authentication-bypass-vulnerabilities.md"
    },
    {
      "source": "data-protection-risk.md",
      "target": "authorization-flaws.md"
    },
    {
      "source": "data-protection-risk.md",
      "target": "buffer-overflow-vulnerabilities.md"
    },
    {
      "source": "data-protection-risk.md",
      "target": "cross-site-scripting-vulnerabilities.md"
    },
    {
      "source": "data-protection-risk.md",
      "target": "insecure-data-transmission.md"
    },
    {
      "source": "data-protection-risk.md",
      "target": "insufficient-audit-logging.md"
    },
    {
      "source": "data-protection-risk.md",
      "target": "log-injection-vulnerabilities.md"
    },
    {
      "source": "data-protection-risk.md",
      "target": "logging-configuration-issues.md"
    },
    {
      "source": "data-protection-risk.md",
      "target": "password-security-weaknesses.md"
    },
    {
      "source": "data-protection-risk.md",
      "target": "secret-management-problems.md"
    },
    {
      "source": "data-protection-risk.md",
      "target": "session-management-issues.md"
    },
    {
      "source": "data-protection-risk.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "data-structure-cache-inefficiency.md",
      "target": "alignment-and-padding-issues.md"
    },
    {
      "source": "data-structure-cache-inefficiency.md",
      "target": "false-sharing.md"
    },
    {
      "source": "database-connection-leaks.md",
      "target": "high-connection-count.md"
    },
    {
      "source": "database-connection-leaks.md",
      "target": "increased-error-rates.md"
    },
    {
      "source": "database-connection-leaks.md",
      "target": "long-running-database-transactions.md"
    },
    {
      "source": "database-connection-leaks.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "database-connection-leaks.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "database-connection-leaks.md",
      "target": "system-outages.md"
    },
    {
      "source": "database-connection-leaks.md",
      "target": "unreleased-resources.md"
    },
    {
      "source": "database-query-performance-issues.md",
      "target": "database-schema-design-problems.md"
    },
    {
      "source": "database-query-performance-issues.md",
      "target": "graphql-complexity-issues.md"
    },
    {
      "source": "database-query-performance-issues.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "database-query-performance-issues.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "database-query-performance-issues.md",
      "target": "incorrect-index-type.md"
    },
    {
      "source": "database-query-performance-issues.md",
      "target": "index-fragmentation.md"
    },
    {
      "source": "database-query-performance-issues.md",
      "target": "inefficient-database-indexing.md"
    },
    {
      "source": "database-query-performance-issues.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "database-query-performance-issues.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "database-query-performance-issues.md",
      "target": "n-plus-one-query-problem.md"
    },
    {
      "source": "database-query-performance-issues.md",
      "target": "queries-that-prevent-index-usage.md"
    },
    {
      "source": "database-query-performance-issues.md",
      "target": "service-timeouts.md"
    },
    {
      "source": "database-query-performance-issues.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "database-schema-design-problems.md",
      "target": "cross-system-data-synchronization-problems.md"
    },
    {
      "source": "database-schema-design-problems.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "database-schema-design-problems.md",
      "target": "data-migration-integrity-issues.md"
    },
    {
      "source": "database-schema-design-problems.md",
      "target": "database-query-performance-issues.md"
    },
    {
      "source": "database-schema-design-problems.md",
      "target": "inefficient-database-indexing.md"
    },
    {
      "source": "database-schema-design-problems.md",
      "target": "long-running-database-transactions.md"
    },
    {
      "source": "database-schema-design-problems.md",
      "target": "poor-domain-model.md"
    },
    {
      "source": "database-schema-design-problems.md",
      "target": "schema-evolution-paralysis.md"
    },
    {
      "source": "database-schema-design-problems.md",
      "target": "unused-indexes.md"
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
      "target": "assumption-based-development.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "database-schema-design-problems.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "hardcoded-values.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "insufficient-audit-logging.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "insufficient-code-review.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "lower-code-quality.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "market-pressure.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "missing-end-to-end-tests.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "poor-test-coverage.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "superficial-code-reviews.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "unrealistic-schedule.md"
    },
    {
      "source": "deadline-pressure.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "deadlock-conditions.md",
      "target": "lock-contention.md"
    },
    {
      "source": "deadlock-conditions.md",
      "target": "long-running-database-transactions.md"
    },
    {
      "source": "deadlock-conditions.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "deadlock-conditions.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "abi-compatibility-issues.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "cache-invalidation-problems.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "circular-dependency-problems.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "delayed-bug-fixes.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "dependency-version-conflicts.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "dma-coherency-issues.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "excessive-logging.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "global-state-and-side-effects.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "hidden-dependencies.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "inadequate-error-handling.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "inadequate-integration-tests.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "insufficient-audit-logging.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "log-injection-vulnerabilities.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "log-spam.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "logging-configuration-issues.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "null-pointer-dereferences.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "race-conditions.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "system-integration-blindness.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "unpredictable-system-behavior.md"
    },
    {
      "source": "debugging-difficulties.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "decision-avoidance.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "decision-avoidance.md",
      "target": "fear-of-failure.md"
    },
    {
      "source": "decision-avoidance.md",
      "target": "work-blocking.md"
    },
    {
      "source": "decision-paralysis.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "decision-paralysis.md",
      "target": "analysis-paralysis.md"
    },
    {
      "source": "decision-paralysis.md",
      "target": "micromanagement-culture.md"
    },
    {
      "source": "decision-paralysis.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "decision-paralysis.md",
      "target": "power-struggles.md"
    },
    {
      "source": "declining-business-metrics.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "declining-business-metrics.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "declining-business-metrics.md",
      "target": "data-protection-risk.md"
    },
    {
      "source": "declining-business-metrics.md",
      "target": "delayed-bug-fixes.md"
    },
    {
      "source": "declining-business-metrics.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "declining-business-metrics.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "declining-business-metrics.md",
      "target": "stakeholder-dissatisfaction.md"
    },
    {
      "source": "declining-business-metrics.md",
      "target": "system-outages.md"
    },
    {
      "source": "defensive-coding-practices.md",
      "target": "unpredictable-system-behavior.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "deadline-pressure.md"
    },
    {
      "source": "delayed-bug-fixes.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "delayed-decision-making.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "delayed-decision-making.md",
      "target": "information-fragmentation.md"
    },
    {
      "source": "delayed-decision-making.md",
      "target": "power-struggles.md"
    },
    {
      "source": "delayed-decision-making.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "delayed-decision-making.md",
      "target": "project-authority-vacuum.md"
    },
    {
      "source": "delayed-issue-resolution.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "delayed-issue-resolution.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "delayed-issue-resolution.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "delayed-issue-resolution.md",
      "target": "procrastination-on-complex-tasks.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "analysis-paralysis.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "changing-project-scope.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "competing-priorities.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "decision-avoidance.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "development-disruption.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "feature-creep.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "merge-conflicts.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "poor-project-control.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "reduced-individual-productivity.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "reduced-team-flexibility.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "reduced-team-productivity.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "scope-creep.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "second-system-effect.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "stakeholder-developer-communication-gap.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "stakeholder-dissatisfaction.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "stakeholder-frustration.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "strangler-fig-pattern-failures.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "unproductive-meetings.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "delayed-project-timelines.md",
      "target": "work-blocking.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "analysis-paralysis.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "dependency-on-supplier.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "extended-review-cycles.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "large-feature-scope.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "long-release-cycles.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "perfectionist-review-culture.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "slow-development-velocity.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "team-coordination-issues.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "team-members-not-engaged-in-review-process.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "technical-architecture-limitations.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "technology-stack-fragmentation.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "test-debt.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "testing-environment-fragility.md"
    },
    {
      "source": "delayed-value-delivery.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "dependency-version-conflicts.md",
      "target": "abi-compatibility-issues.md"
    },
    {
      "source": "dependency-version-conflicts.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "deployment-coupling.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "deployment-coupling.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "deployment-coupling.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "deployment-coupling.md",
      "target": "shared-database.md"
    },
    {
      "source": "deployment-coupling.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "deployment-coupling.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "deployment-environment-inconsistencies.md",
      "target": "abi-compatibility-issues.md"
    },
    {
      "source": "deployment-environment-inconsistencies.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "deployment-environment-inconsistencies.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "deployment-environment-inconsistencies.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "deployment-environment-inconsistencies.md",
      "target": "dependency-version-conflicts.md"
    },
    {
      "source": "deployment-environment-inconsistencies.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "deployment-environment-inconsistencies.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "deployment-environment-inconsistencies.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "deployment-environment-inconsistencies.md",
      "target": "hardcoded-values.md"
    },
    {
      "source": "deployment-environment-inconsistencies.md",
      "target": "inadequate-configuration-management.md"
    },
    {
      "source": "deployment-environment-inconsistencies.md",
      "target": "legacy-configuration-management-chaos.md"
    },
    {
      "source": "deployment-environment-inconsistencies.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "deployment-environment-inconsistencies.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "deployment-environment-inconsistencies.md",
      "target": "secret-management-problems.md"
    },
    {
      "source": "deployment-risk.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "deployment-risk.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "deployment-risk.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "deployment-risk.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "deployment-risk.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "deployment-risk.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "deployment-risk.md",
      "target": "monolithic-architecture-constraints.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "author-frustration.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "capacity-mismatch.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "cascade-delays.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "complex-implementation-paths.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "development-disruption.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "feature-factory.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "frequent-changes-to-requirements.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "high-turnover.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "implementation-rework.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "inappropriate-skillset.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "increased-manual-work.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "inefficient-development-environment.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "invisible-nature-of-technical-debt.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "lower-code-quality.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "market-pressure.md"
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
      "target": "merge-conflicts.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "micromanagement-culture.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "new-hire-frustration.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "nitpicking-culture.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "no-formal-change-control-process.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "poor-communication.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "priority-thrashing.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "reduced-individual-productivity.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "slow-development-velocity.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "stakeholder-frustration.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "team-demoralization.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "testing-environment-fragility.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "unrealistic-schedule.md"
    },
    {
      "source": "developer-frustration-and-burnout.md",
      "target": "work-blocking.md"
    },
    {
      "source": "development-disruption.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "development-disruption.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "development-disruption.md",
      "target": "developer-frustration-and-burnout.md"
    },
    {
      "source": "development-disruption.md",
      "target": "priority-thrashing.md"
    },
    {
      "source": "development-disruption.md",
      "target": "reduced-individual-productivity.md"
    },
    {
      "source": "development-disruption.md",
      "target": "reduced-team-flexibility.md"
    },
    {
      "source": "development-disruption.md",
      "target": "reduced-team-productivity.md"
    },
    {
      "source": "development-disruption.md",
      "target": "release-instability.md"
    },
    {
      "source": "development-disruption.md",
      "target": "unproductive-meetings.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "bloated-class.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "clever-code.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "difficult-developer-onboarding.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "high-coupling-low-cohesion.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "inconsistent-codebase.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "inconsistent-coding-standards.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "inconsistent-naming-conventions.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "monolithic-functions-and-classes.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "new-hire-frustration.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "procedural-programming-in-oop-languages.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "procrastination-on-complex-tasks.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "difficult-code-comprehension.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "difficult-code-reuse.md",
      "target": "circular-dependency-problems.md"
    },
    {
      "source": "difficult-code-reuse.md",
      "target": "code-duplication.md"
    },
    {
      "source": "difficult-code-reuse.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "difficult-code-reuse.md",
      "target": "hardcoded-values.md"
    },
    {
      "source": "difficult-code-reuse.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "difficult-code-reuse.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "difficult-code-reuse.md",
      "target": "monolithic-functions-and-classes.md"
    },
    {
      "source": "difficult-code-reuse.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "difficult-code-reuse.md",
      "target": "procedural-background.md"
    },
    {
      "source": "difficult-code-reuse.md",
      "target": "procedural-programming-in-oop-languages.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "complex-domain-model.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "high-turnover.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "inconsistent-codebase.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "inconsistent-coding-standards.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "inconsistent-naming-conventions.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "inefficient-development-environment.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "inexperienced-developers.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "information-decay.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "information-fragmentation.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "knowledge-silos.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "legacy-system-documentation-archaeology.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "new-hire-frustration.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "obsolete-technologies.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "poor-communication.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "technology-stack-fragmentation.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "unclear-documentation-ownership.md"
    },
    {
      "source": "difficult-developer-onboarding.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "circular-dependency-problems.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "flaky-tests.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "global-state-and-side-effects.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "high-coupling-low-cohesion.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "legacy-code-without-tests.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "missing-end-to-end-tests.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "monolithic-functions-and-classes.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "poor-test-coverage.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "difficult-to-test-code.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "difficult-to-understand-code.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "difficult-to-understand-code.md",
      "target": "increased-risk-of-bugs.md"
    },
    {
      "source": "difficulty-quantifying-benefits.md",
      "target": "invisible-nature-of-technical-debt.md"
    },
    {
      "source": "difficulty-quantifying-benefits.md",
      "target": "modernization-roi-justification-failure.md"
    },
    {
      "source": "difficulty-quantifying-benefits.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "difficulty-quantifying-benefits.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "duplicated-effort.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "duplicated-effort.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "duplicated-effort.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "duplicated-effort.md",
      "target": "poor-communication.md"
    },
    {
      "source": "duplicated-effort.md",
      "target": "team-confusion.md"
    },
    {
      "source": "duplicated-effort.md",
      "target": "team-coordination-issues.md"
    },
    {
      "source": "duplicated-effort.md",
      "target": "team-silos.md"
    },
    {
      "source": "duplicated-research-effort.md",
      "target": "knowledge-silos.md"
    },
    {
      "source": "duplicated-work.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "duplicated-work.md",
      "target": "language-barriers.md"
    },
    {
      "source": "duplicated-work.md",
      "target": "unclear-documentation-ownership.md"
    },
    {
      "source": "duplicated-work.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "eager-to-please-stakeholders.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "eager-to-please-stakeholders.md",
      "target": "feature-bloat.md"
    },
    {
      "source": "eager-to-please-stakeholders.md",
      "target": "feature-creep.md"
    },
    {
      "source": "eager-to-please-stakeholders.md",
      "target": "gold-plating.md"
    },
    {
      "source": "eager-to-please-stakeholders.md",
      "target": "market-pressure.md"
    },
    {
      "source": "eager-to-please-stakeholders.md",
      "target": "no-formal-change-control-process.md"
    },
    {
      "source": "eager-to-please-stakeholders.md",
      "target": "planning-credibility-issues.md"
    },
    {
      "source": "eager-to-please-stakeholders.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "eager-to-please-stakeholders.md",
      "target": "scope-creep.md"
    },
    {
      "source": "excessive-class-size.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "excessive-disk-io.md",
      "target": "excessive-logging.md"
    },
    {
      "source": "excessive-disk-io.md",
      "target": "log-spam.md"
    },
    {
      "source": "excessive-disk-io.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "excessive-disk-io.md",
      "target": "unbounded-data-growth.md"
    },
    {
      "source": "excessive-disk-io.md",
      "target": "unoptimized-file-access.md"
    },
    {
      "source": "excessive-disk-io.md",
      "target": "virtual-memory-thrashing.md"
    },
    {
      "source": "excessive-logging.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "excessive-logging.md",
      "target": "log-spam.md"
    },
    {
      "source": "excessive-logging.md",
      "target": "unbounded-data-growth.md"
    },
    {
      "source": "excessive-object-allocation.md",
      "target": "garbage-collection-pressure.md"
    },
    {
      "source": "excessive-object-allocation.md",
      "target": "memory-fragmentation.md"
    },
    {
      "source": "excessive-object-allocation.md",
      "target": "memory-leaks.md"
    },
    {
      "source": "excessive-object-allocation.md",
      "target": "serialization-deserialization-bottlenecks.md"
    },
    {
      "source": "extended-cycle-times.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "extended-cycle-times.md",
      "target": "extended-review-cycles.md"
    },
    {
      "source": "extended-cycle-times.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "extended-cycle-times.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "extended-research-time.md",
      "target": "complex-domain-model.md"
    },
    {
      "source": "extended-research-time.md",
      "target": "duplicated-research-effort.md"
    },
    {
      "source": "extended-research-time.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "extended-research-time.md",
      "target": "legacy-system-documentation-archaeology.md"
    },
    {
      "source": "extended-research-time.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "extended-review-cycles.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "extended-review-cycles.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "extended-review-cycles.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "extended-review-cycles.md",
      "target": "inadequate-initial-reviews.md"
    },
    {
      "source": "extended-review-cycles.md",
      "target": "perfectionist-review-culture.md"
    },
    {
      "source": "extended-review-cycles.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "external-service-delays.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "external-service-delays.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "external-service-delays.md",
      "target": "long-running-database-transactions.md"
    },
    {
      "source": "external-service-delays.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "external-service-delays.md",
      "target": "network-latency.md"
    },
    {
      "source": "external-service-delays.md",
      "target": "service-timeouts.md"
    },
    {
      "source": "external-service-delays.md",
      "target": "upstream-timeouts.md"
    },
    {
      "source": "false-sharing.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "false-sharing.md",
      "target": "dma-coherency-issues.md"
    },
    {
      "source": "fear-of-breaking-changes.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "fear-of-breaking-changes.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "fear-of-breaking-changes.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "fear-of-breaking-changes.md",
      "target": "fear-of-failure.md"
    },
    {
      "source": "fear-of-breaking-changes.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "fear-of-breaking-changes.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "fear-of-breaking-changes.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "fear-of-breaking-changes.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "fear-of-breaking-changes.md",
      "target": "ripple-effect-of-changes.md"
    },
    {
      "source": "fear-of-breaking-changes.md",
      "target": "schema-evolution-paralysis.md"
    },
    {
      "source": "fear-of-breaking-changes.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "clever-code.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "difficult-to-test-code.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "fear-of-failure.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "hidden-dependencies.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "high-coupling-low-cohesion.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "high-defect-rate-in-production.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "inability-to-innovate.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "increased-risk-of-bugs.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "legacy-business-logic-extraction-difficulty.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "legacy-code-without-tests.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "micromanagement-culture.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "monolithic-functions-and-classes.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "poor-test-coverage.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "reduced-code-submission-frequency.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "regression-bugs.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "regulatory-compliance-drift.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "stagnant-architecture.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "test-debt.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "unpredictable-system-behavior.md"
    },
    {
      "source": "fear-of-change.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "fear-of-conflict.md",
      "target": "decision-avoidance.md"
    },
    {
      "source": "fear-of-conflict.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "fear-of-conflict.md",
      "target": "poor-communication.md"
    },
    {
      "source": "fear-of-conflict.md",
      "target": "reviewer-anxiety.md"
    },
    {
      "source": "fear-of-conflict.md",
      "target": "superficial-code-reviews.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "analysis-paralysis.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "blame-culture.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "perfectionist-review-culture.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "poor-project-control.md"
    },
    {
      "source": "fear-of-failure.md",
      "target": "reduced-innovation.md"
    },
    {
      "source": "feature-bloat.md",
      "target": "feature-creep.md"
    },
    {
      "source": "feature-bloat.md",
      "target": "feature-factory.md"
    },
    {
      "source": "feature-bloat.md",
      "target": "gold-plating.md"
    },
    {
      "source": "feature-bloat.md",
      "target": "scope-creep.md"
    },
    {
      "source": "feature-bloat.md",
      "target": "second-system-effect.md"
    },
    {
      "source": "feature-creep-without-refactoring.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "feature-creep-without-refactoring.md",
      "target": "database-schema-design-problems.md"
    },
    {
      "source": "feature-creep-without-refactoring.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "feature-creep-without-refactoring.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "feature-creep-without-refactoring.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "feature-creep-without-refactoring.md",
      "target": "high-coupling-low-cohesion.md"
    },
    {
      "source": "feature-creep-without-refactoring.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "feature-creep-without-refactoring.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "feature-creep-without-refactoring.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "feature-creep-without-refactoring.md",
      "target": "stagnant-architecture.md"
    },
    {
      "source": "feature-creep-without-refactoring.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "feature-creep-without-refactoring.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "feature-creep-without-refactoring.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "feature-creep.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "feature-creep.md",
      "target": "bloated-class.md"
    },
    {
      "source": "feature-creep.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "feature-creep.md",
      "target": "feature-bloat.md"
    },
    {
      "source": "feature-creep.md",
      "target": "large-feature-scope.md"
    },
    {
      "source": "feature-creep.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "feature-factory.md",
      "target": "delayed-bug-fixes.md"
    },
    {
      "source": "feature-factory.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "feature-factory.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "feature-factory.md",
      "target": "feature-bloat.md"
    },
    {
      "source": "feature-factory.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "feature-factory.md",
      "target": "invisible-nature-of-technical-debt.md"
    },
    {
      "source": "feature-gaps.md",
      "target": "assumption-based-development.md"
    },
    {
      "source": "feature-gaps.md",
      "target": "feedback-isolation.md"
    },
    {
      "source": "feature-gaps.md",
      "target": "inadequate-requirements-gathering.md"
    },
    {
      "source": "feature-gaps.md",
      "target": "negative-user-feedback.md"
    },
    {
      "source": "feature-gaps.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "assumption-based-development.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "communication-risk-outside-project.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "feature-bloat.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "feature-factory.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "feature-gaps.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "inadequate-requirements-gathering.md"
    },
    {
      "source": "feedback-isolation.md",
      "target": "stakeholder-developer-communication-gap.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "inadequate-test-data-management.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "inadequate-test-infrastructure.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "flaky-tests.md",
      "target": "testing-environment-fragility.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "development-disruption.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "feature-creep.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "frequent-changes-to-requirements.md",
      "target": "team-confusion.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "high-defect-rate-in-production.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "inadequate-configuration-management.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "inadequate-integration-tests.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "release-instability.md"
    },
    {
      "source": "frequent-hotfixes-and-rollbacks.md",
      "target": "system-integration-blindness.md"
    },
    {
      "source": "garbage-collection-pressure.md",
      "target": "circular-references.md"
    },
    {
      "source": "garbage-collection-pressure.md",
      "target": "excessive-object-allocation.md"
    },
    {
      "source": "garbage-collection-pressure.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "global-state-and-side-effects.md",
      "target": "difficult-to-test-code.md"
    },
    {
      "source": "global-state-and-side-effects.md",
      "target": "hidden-dependencies.md"
    },
    {
      "source": "global-state-and-side-effects.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "global-state-and-side-effects.md",
      "target": "lock-contention.md"
    },
    {
      "source": "god-object-anti-pattern.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "god-object-anti-pattern.md",
      "target": "difficult-to-test-code.md"
    },
    {
      "source": "god-object-anti-pattern.md",
      "target": "lock-contention.md"
    },
    {
      "source": "god-object-anti-pattern.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "god-object-anti-pattern.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "god-object-anti-pattern.md",
      "target": "ripple-effect-of-changes.md"
    },
    {
      "source": "god-object-anti-pattern.md",
      "target": "single-entry-point-design.md"
    },
    {
      "source": "gold-plating.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "gold-plating.md",
      "target": "second-system-effect.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "circular-references.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "data-structure-cache-inefficiency.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "dma-coherency-issues.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "excessive-logging.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "excessive-object-allocation.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "false-sharing.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "garbage-collection-pressure.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "incorrect-index-type.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "insufficient-worker-capacity.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "memory-barrier-inefficiency.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "memory-fragmentation.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "memory-leaks.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "queries-that-prevent-index-usage.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "resource-contention.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "system-outages.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "unbounded-data-growth.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "unbounded-data-structures.md"
    },
    {
      "source": "gradual-performance-degradation.md",
      "target": "unoptimized-file-access.md"
    },
    {
      "source": "growing-task-queues.md",
      "target": "insufficient-worker-capacity.md"
    },
    {
      "source": "hardcoded-values.md",
      "target": "legacy-configuration-management-chaos.md"
    },
    {
      "source": "hardcoded-values.md",
      "target": "secret-management-problems.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "dependency-version-conflicts.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "global-state-and-side-effects.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "ripple-effect-of-changes.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "strangler-fig-pattern-failures.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "system-integration-blindness.md"
    },
    {
      "source": "hidden-dependencies.md",
      "target": "unpredictable-system-behavior.md"
    },
    {
      "source": "high-api-latency.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "high-api-latency.md",
      "target": "garbage-collection-pressure.md"
    },
    {
      "source": "high-api-latency.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "high-api-latency.md",
      "target": "graphql-complexity-issues.md"
    },
    {
      "source": "high-api-latency.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "high-api-latency.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "high-api-latency.md",
      "target": "network-latency.md"
    },
    {
      "source": "high-api-latency.md",
      "target": "poor-caching-strategy.md"
    },
    {
      "source": "high-api-latency.md",
      "target": "serialization-deserialization-bottlenecks.md"
    },
    {
      "source": "high-api-latency.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "high-api-latency.md",
      "target": "slow-database-queries.md"
    },
    {
      "source": "high-api-latency.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "high-api-latency.md",
      "target": "upstream-timeouts.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "complex-domain-model.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "deadline-pressure.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "difficult-to-test-code.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "fear-of-conflict.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "hidden-dependencies.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "high-coupling-low-cohesion.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "high-turnover.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "inappropriate-skillset.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "increased-error-rates.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "increased-risk-of-bugs.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "insufficient-code-review.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "lower-code-quality.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "monolithic-functions-and-classes.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "high-bug-introduction-rate.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "high-client-side-resource-consumption.md",
      "target": "circular-references.md"
    },
    {
      "source": "high-client-side-resource-consumption.md",
      "target": "endianness-conversion-overhead.md"
    },
    {
      "source": "high-client-side-resource-consumption.md",
      "target": "excessive-object-allocation.md"
    },
    {
      "source": "high-client-side-resource-consumption.md",
      "target": "inefficient-frontend-code.md"
    },
    {
      "source": "high-client-side-resource-consumption.md",
      "target": "memory-leaks.md"
    },
    {
      "source": "high-client-side-resource-consumption.md",
      "target": "rate-limiting-issues.md"
    },
    {
      "source": "high-connection-count.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "high-connection-count.md",
      "target": "database-query-performance-issues.md"
    },
    {
      "source": "high-connection-count.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "high-connection-count.md",
      "target": "incorrect-max-connection-pool-size.md"
    },
    {
      "source": "high-connection-count.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "high-connection-count.md",
      "target": "upstream-timeouts.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "difficult-to-test-code.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "legacy-business-logic-extraction-difficulty.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "legacy-code-without-tests.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "lock-contention.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "merge-conflicts.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "microservice-communication-overhead.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "single-entry-point-design.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "strangler-fig-pattern-failures.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "high-coupling-low-cohesion.md",
      "target": "unpredictable-system-behavior.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "database-query-performance-issues.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "graphql-complexity-issues.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "high-connection-count.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "incorrect-index-type.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "incorrect-max-connection-pool-size.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "index-fragmentation.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "inefficient-database-indexing.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "log-spam.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "long-running-database-transactions.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "high-database-resource-utilization.md",
      "target": "poor-caching-strategy.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "development-disruption.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "inadequate-code-reviews.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "inadequate-integration-tests.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "insufficient-code-review.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "large-pull-requests.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "legacy-code-without-tests.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "poor-test-coverage.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "review-process-avoidance.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "review-process-breakdown.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "reviewer-inexperience.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "stakeholder-dissatisfaction.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "stakeholder-frustration.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "superficial-code-reviews.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "system-integration-blindness.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "test-debt.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "high-defect-rate-in-production.md",
      "target": "testing-environment-fragility.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "cv-driven-development.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "dependency-on-supplier.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "feature-bloat.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "high-defect-rate-in-production.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "inability-to-innovate.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "obsolete-technologies.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "ripple-effect-of-changes.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "strangler-fig-pattern-failures.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "technical-architecture-limitations.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "technology-isolation.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "technology-stack-fragmentation.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "high-maintenance-costs.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "high-number-of-database-queries.md",
      "target": "database-schema-design-problems.md"
    },
    {
      "source": "high-number-of-database-queries.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "high-number-of-database-queries.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "high-number-of-database-queries.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "high-number-of-database-queries.md",
      "target": "n-plus-one-query-problem.md"
    },
    {
      "source": "high-number-of-database-queries.md",
      "target": "poor-caching-strategy.md"
    },
    {
      "source": "high-number-of-database-queries.md",
      "target": "slow-database-queries.md"
    },
    {
      "source": "high-number-of-database-queries.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "high-resource-utilization-on-client.md",
      "target": "improper-event-listener-management.md"
    },
    {
      "source": "high-resource-utilization-on-client.md",
      "target": "inefficient-frontend-code.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "bloated-class.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "complex-implementation-paths.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "deadline-pressure.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "delayed-bug-fixes.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "developer-frustration-and-burnout.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "feature-creep.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "feature-factory.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "high-defect-rate-in-production.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "high-turnover.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "implementation-starts-without-design.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "inability-to-innovate.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "inadequate-code-reviews.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "insufficient-code-review.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "invisible-nature-of-technical-debt.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "lower-code-quality.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "maintenance-cost-increase.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "modernization-roi-justification-failure.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "reduced-innovation.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "review-process-breakdown.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "slow-development-velocity.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "stagnant-architecture.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "superficial-code-reviews.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "time-pressure.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "high-technical-debt.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "high-turnover.md",
      "target": "blame-culture.md"
    },
    {
      "source": "high-turnover.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "high-turnover.md",
      "target": "cv-driven-development.md"
    },
    {
      "source": "high-turnover.md",
      "target": "developer-frustration-and-burnout.md"
    },
    {
      "source": "high-turnover.md",
      "target": "difficult-developer-onboarding.md"
    },
    {
      "source": "high-turnover.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "high-turnover.md",
      "target": "inability-to-innovate.md"
    },
    {
      "source": "high-turnover.md",
      "target": "inadequate-mentoring-structure.md"
    },
    {
      "source": "high-turnover.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "high-turnover.md",
      "target": "inconsistent-coding-standards.md"
    },
    {
      "source": "high-turnover.md",
      "target": "inconsistent-onboarding-experience.md"
    },
    {
      "source": "high-turnover.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "high-turnover.md",
      "target": "inexperienced-developers.md"
    },
    {
      "source": "high-turnover.md",
      "target": "information-decay.md"
    },
    {
      "source": "high-turnover.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "high-turnover.md",
      "target": "legacy-skill-shortage.md"
    },
    {
      "source": "high-turnover.md",
      "target": "legacy-system-documentation-archaeology.md"
    },
    {
      "source": "high-turnover.md",
      "target": "micromanagement-culture.md"
    },
    {
      "source": "high-turnover.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "high-turnover.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "high-turnover.md",
      "target": "new-hire-frustration.md"
    },
    {
      "source": "high-turnover.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "high-turnover.md",
      "target": "poor-teamwork.md"
    },
    {
      "source": "high-turnover.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "high-turnover.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "high-turnover.md",
      "target": "team-churn-impact.md"
    },
    {
      "source": "high-turnover.md",
      "target": "team-demoralization.md"
    },
    {
      "source": "high-turnover.md",
      "target": "team-dysfunction.md"
    },
    {
      "source": "history-of-failed-changes.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "history-of-failed-changes.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "history-of-failed-changes.md",
      "target": "fear-of-failure.md"
    },
    {
      "source": "history-of-failed-changes.md",
      "target": "high-defect-rate-in-production.md"
    },
    {
      "source": "history-of-failed-changes.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "history-of-failed-changes.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "history-of-failed-changes.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "history-of-failed-changes.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "immature-delivery-strategy.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "immature-delivery-strategy.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "immature-delivery-strategy.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "imperative-data-fetching-logic.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "imperative-data-fetching-logic.md",
      "target": "n-plus-one-query-problem.md"
    },
    {
      "source": "imperative-data-fetching-logic.md",
      "target": "slow-database-queries.md"
    },
    {
      "source": "imperative-data-fetching-logic.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "assumption-based-development.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "changing-project-scope.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "duplicated-work.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "feedback-isolation.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "frequent-changes-to-requirements.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "implementation-starts-without-design.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "inadequate-requirements-gathering.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "poor-communication.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "poor-planning.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "implementation-rework.md",
      "target": "stakeholder-developer-communication-gap.md"
    },
    {
      "source": "implementation-starts-without-design.md",
      "target": "circular-dependency-problems.md"
    },
    {
      "source": "implementation-starts-without-design.md",
      "target": "database-schema-design-problems.md"
    },
    {
      "source": "implementation-starts-without-design.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "implementation-starts-without-design.md",
      "target": "implementation-rework.md"
    },
    {
      "source": "implementation-starts-without-design.md",
      "target": "poor-caching-strategy.md"
    },
    {
      "source": "implementation-starts-without-design.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "implementation-starts-without-design.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "implicit-knowledge.md",
      "target": "difficult-developer-onboarding.md"
    },
    {
      "source": "implicit-knowledge.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "implicit-knowledge.md",
      "target": "high-turnover.md"
    },
    {
      "source": "implicit-knowledge.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "implicit-knowledge.md",
      "target": "knowledge-silos.md"
    },
    {
      "source": "implicit-knowledge.md",
      "target": "legacy-business-logic-extraction-difficulty.md"
    },
    {
      "source": "implicit-knowledge.md",
      "target": "legacy-system-documentation-archaeology.md"
    },
    {
      "source": "implicit-knowledge.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "implicit-knowledge.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "improper-event-listener-management.md",
      "target": "circular-references.md"
    },
    {
      "source": "improper-event-listener-management.md",
      "target": "high-client-side-resource-consumption.md"
    },
    {
      "source": "improper-event-listener-management.md",
      "target": "high-resource-utilization-on-client.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "inability-to-innovate.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "buffer-overflow-vulnerabilities.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "cross-site-scripting-vulnerabilities.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "fear-of-conflict.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "inadequate-error-handling.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "inadequate-initial-reviews.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "inconsistent-codebase.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "reduced-code-submission-frequency.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "reduced-review-participation.md"
    },
    {
      "source": "inadequate-code-reviews.md",
      "target": "regression-bugs.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "incorrect-max-connection-pool-size.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "logging-configuration-issues.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "secret-management-problems.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "testing-environment-fragility.md"
    },
    {
      "source": "inadequate-configuration-management.md",
      "target": "unbounded-data-growth.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "authentication-bypass-vulnerabilities.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "authorization-flaws.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "error-message-information-disclosure.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "excessive-logging.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "increased-customer-support-load.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "increased-error-rates.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "integer-overflow-underflow.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "log-injection-vulnerabilities.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "log-spam.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "null-pointer-dereferences.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "system-outages.md"
    },
    {
      "source": "inadequate-error-handling.md",
      "target": "unreleased-resources.md"
    },
    {
      "source": "inadequate-initial-reviews.md",
      "target": "extended-review-cycles.md"
    },
    {
      "source": "inadequate-initial-reviews.md",
      "target": "inadequate-code-reviews.md"
    },
    {
      "source": "inadequate-integration-tests.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "inadequate-integration-tests.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "inadequate-integration-tests.md",
      "target": "high-defect-rate-in-production.md"
    },
    {
      "source": "inadequate-integration-tests.md",
      "target": "system-integration-blindness.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "fear-of-conflict.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "inappropriate-skillset.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "inconsistent-knowledge-acquisition.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "inconsistent-onboarding-experience.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "inexperienced-developers.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "mentor-burnout.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "reviewer-anxiety.md"
    },
    {
      "source": "inadequate-mentoring-structure.md",
      "target": "reviewer-inexperience.md"
    },
    {
      "source": "inadequate-onboarding.md",
      "target": "difficult-developer-onboarding.md"
    },
    {
      "source": "inadequate-onboarding.md",
      "target": "mentor-burnout.md"
    },
    {
      "source": "inadequate-onboarding.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "inadequate-onboarding.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "inadequate-onboarding.md",
      "target": "team-churn-impact.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "assumption-based-development.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "changing-project-scope.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "feature-gaps.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "frequent-changes-to-requirements.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "gold-plating.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "inadequate-error-handling.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "inadequate-onboarding.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "large-feature-scope.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "poor-contract-design.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "poor-planning.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "rest-api-design-issues.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "scope-creep.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "stakeholder-developer-communication-gap.md"
    },
    {
      "source": "inadequate-requirements-gathering.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "flaky-tests.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "inadequate-integration-tests.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "inadequate-test-data-management.md",
      "target": "testing-environment-fragility.md"
    },
    {
      "source": "inadequate-test-infrastructure.md",
      "target": "inadequate-integration-tests.md"
    },
    {
      "source": "inadequate-test-infrastructure.md",
      "target": "inadequate-test-data-management.md"
    },
    {
      "source": "inadequate-test-infrastructure.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "inadequate-test-infrastructure.md",
      "target": "increased-manual-work.md"
    },
    {
      "source": "inadequate-test-infrastructure.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "inadequate-test-infrastructure.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "inadequate-test-infrastructure.md",
      "target": "poor-test-coverage.md"
    },
    {
      "source": "inadequate-test-infrastructure.md",
      "target": "testing-environment-fragility.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "inconsistent-knowledge-acquisition.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "inconsistent-onboarding-experience.md"
    },
    {
      "source": "inappropriate-skillset.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "incomplete-knowledge.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "incomplete-knowledge.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "incomplete-knowledge.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "incomplete-knowledge.md",
      "target": "incorrect-index-type.md"
    },
    {
      "source": "incomplete-knowledge.md",
      "target": "incorrect-max-connection-pool-size.md"
    },
    {
      "source": "incomplete-knowledge.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "incomplete-knowledge.md",
      "target": "poor-domain-model.md"
    },
    {
      "source": "incomplete-knowledge.md",
      "target": "poor-planning.md"
    },
    {
      "source": "incomplete-knowledge.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "incomplete-knowledge.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "incomplete-projects.md",
      "target": "competing-priorities.md"
    },
    {
      "source": "incomplete-projects.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "incomplete-projects.md",
      "target": "poor-project-control.md"
    },
    {
      "source": "incomplete-projects.md",
      "target": "reduced-team-productivity.md"
    },
    {
      "source": "incomplete-projects.md",
      "target": "scope-creep.md"
    },
    {
      "source": "incomplete-projects.md",
      "target": "stakeholder-dissatisfaction.md"
    },
    {
      "source": "incomplete-projects.md",
      "target": "unclear-goals-and-priorities.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "cache-invalidation-problems.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "code-duplication.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "cross-system-data-synchronization-problems.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "duplicated-work.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "hardcoded-values.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "inconsistent-codebase.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "user-confusion.md"
    },
    {
      "source": "inconsistent-behavior.md",
      "target": "user-frustration.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "inadequate-code-reviews.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "insufficient-code-review.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "poor-teamwork.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "team-churn-impact.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "team-coordination-issues.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "team-silos.md"
    },
    {
      "source": "inconsistent-codebase.md",
      "target": "undefined-code-style-guidelines.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "author-frustration.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "bloated-class.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "buffer-overflow-vulnerabilities.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "improper-event-listener-management.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "rapid-team-growth.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "rest-api-design-issues.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "review-process-avoidance.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "review-process-breakdown.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "sql-injection-vulnerabilities.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "team-churn-impact.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "team-members-not-engaged-in-review-process.md"
    },
    {
      "source": "inconsistent-coding-standards.md",
      "target": "unreleased-resources.md"
    },
    {
      "source": "inconsistent-execution.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "inconsistent-execution.md",
      "target": "increased-manual-work.md"
    },
    {
      "source": "inconsistent-knowledge-acquisition.md",
      "target": "inadequate-mentoring-structure.md"
    },
    {
      "source": "inconsistent-knowledge-acquisition.md",
      "target": "inconsistent-onboarding-experience.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "undefined-code-style-guidelines.md"
    },
    {
      "source": "inconsistent-naming-conventions.md",
      "target": "user-confusion.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "inadequate-mentoring-structure.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "inconsistent-knowledge-acquisition.md"
    },
    {
      "source": "inconsistent-onboarding-experience.md",
      "target": "new-hire-frustration.md"
    },
    {
      "source": "inconsistent-quality.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "inconsistent-quality.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "inconsistent-quality.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "inconsistent-quality.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "inconsistent-quality.md",
      "target": "quality-compromises.md"
    },
    {
      "source": "inconsistent-quality.md",
      "target": "superficial-code-reviews.md"
    },
    {
      "source": "inconsistent-quality.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "incorrect-index-type.md",
      "target": "inefficient-database-indexing.md"
    },
    {
      "source": "incorrect-max-connection-pool-size.md",
      "target": "high-connection-count.md"
    },
    {
      "source": "incorrect-max-connection-pool-size.md",
      "target": "increased-error-rates.md"
    },
    {
      "source": "incorrect-max-connection-pool-size.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "incorrect-max-connection-pool-size.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "increased-bug-count.md",
      "target": "development-disruption.md"
    },
    {
      "source": "increased-bug-count.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "increased-bug-count.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "increased-bug-count.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "increased-bug-count.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "increased-bug-count.md",
      "target": "increased-customer-support-load.md"
    },
    {
      "source": "increased-bug-count.md",
      "target": "increased-risk-of-bugs.md"
    },
    {
      "source": "increased-bug-count.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "increased-bug-count.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "increased-bug-count.md",
      "target": "large-pull-requests.md"
    },
    {
      "source": "increased-bug-count.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "increased-bug-count.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "increased-bug-count.md",
      "target": "regression-bugs.md"
    },
    {
      "source": "increased-bug-count.md",
      "target": "superficial-code-reviews.md"
    },
    {
      "source": "increased-cognitive-load.md",
      "target": "clever-code.md"
    },
    {
      "source": "increased-cognitive-load.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "increased-cognitive-load.md",
      "target": "defensive-coding-practices.md"
    },
    {
      "source": "increased-cognitive-load.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "increased-cognitive-load.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "increased-cognitive-load.md",
      "target": "feature-bloat.md"
    },
    {
      "source": "increased-cognitive-load.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "increased-cognitive-load.md",
      "target": "high-coupling-low-cohesion.md"
    },
    {
      "source": "increased-cognitive-load.md",
      "target": "inconsistent-codebase.md"
    },
    {
      "source": "increased-cognitive-load.md",
      "target": "inconsistent-coding-standards.md"
    },
    {
      "source": "increased-cognitive-load.md",
      "target": "inconsistent-naming-conventions.md"
    },
    {
      "source": "increased-cognitive-load.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "increased-cognitive-load.md",
      "target": "mental-fatigue.md"
    },
    {
      "source": "increased-cognitive-load.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "complex-implementation-paths.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "feature-creep.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "gold-plating.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "hardcoded-values.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "high-turnover.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "implementation-rework.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "increased-risk-of-bugs.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "integration-difficulties.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "legacy-api-versioning-nightmare.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "legacy-business-logic-extraction-difficulty.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "legacy-skill-shortage.md"
    },
    {
      "source": "increased-cost-of-development.md",
      "target": "second-system-effect.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "delayed-bug-fixes.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "feature-gaps.md"
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
      "target": "increased-error-rates.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "increased-customer-support-load.md",
      "target": "system-outages.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "abi-compatibility-issues.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "buffer-overflow-vulnerabilities.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "cache-invalidation-problems.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "data-migration-integrity-issues.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "dependency-version-conflicts.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "dma-coherency-issues.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "high-connection-count.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "inadequate-error-handling.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "incorrect-max-connection-pool-size.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "load-balancing-problems.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "mental-fatigue.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "merge-conflicts.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "missing-end-to-end-tests.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "null-pointer-dereferences.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "race-conditions.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "unrealistic-schedule.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "upstream-timeouts.md"
    },
    {
      "source": "increased-error-rates.md",
      "target": "user-trust-erosion.md"
    },
    {
      "source": "increased-manual-testing-effort.md",
      "target": "difficult-to-test-code.md"
    },
    {
      "source": "increased-manual-testing-effort.md",
      "target": "flaky-tests.md"
    },
    {
      "source": "increased-manual-testing-effort.md",
      "target": "inadequate-test-data-management.md"
    },
    {
      "source": "increased-manual-testing-effort.md",
      "target": "inadequate-test-infrastructure.md"
    },
    {
      "source": "increased-manual-testing-effort.md",
      "target": "legacy-code-without-tests.md"
    },
    {
      "source": "increased-manual-testing-effort.md",
      "target": "long-release-cycles.md"
    },
    {
      "source": "increased-manual-testing-effort.md",
      "target": "poor-test-coverage.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "cross-system-data-synchronization-problems.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "data-migration-integrity-issues.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "legacy-configuration-management-chaos.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "reduced-individual-productivity.md"
    },
    {
      "source": "increased-manual-work.md",
      "target": "regulatory-compliance-drift.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "code-duplication.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "context-switching-overhead.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "database-schema-design-problems.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "inadequate-code-reviews.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "inconsistent-coding-standards.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "inconsistent-naming-conventions.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "inexperienced-developers.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "information-decay.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "insufficient-code-review.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "increased-risk-of-bugs.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "increased-stress-and-burnout.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "increased-stress-and-burnout.md",
      "target": "deadline-pressure.md"
    },
    {
      "source": "increased-stress-and-burnout.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "increased-stress-and-burnout.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "increased-stress-and-burnout.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "increased-stress-and-burnout.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "increased-stress-and-burnout.md",
      "target": "poor-planning.md"
    },
    {
      "source": "increased-stress-and-burnout.md",
      "target": "scope-creep.md"
    },
    {
      "source": "increased-stress-and-burnout.md",
      "target": "time-pressure.md"
    },
    {
      "source": "increased-stress-and-burnout.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "increased-stress-and-burnout.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "increased-stress-and-burnout.md",
      "target": "unmotivated-employees.md"
    },
    {
      "source": "increased-technical-shortcuts.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "increased-technical-shortcuts.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "increased-technical-shortcuts.md",
      "target": "deadline-pressure.md"
    },
    {
      "source": "increased-technical-shortcuts.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "increased-technical-shortcuts.md",
      "target": "inexperienced-developers.md"
    },
    {
      "source": "increased-technical-shortcuts.md",
      "target": "invisible-nature-of-technical-debt.md"
    },
    {
      "source": "increased-technical-shortcuts.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "increased-technical-shortcuts.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "increased-technical-shortcuts.md",
      "target": "quality-compromises.md"
    },
    {
      "source": "increased-technical-shortcuts.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "increased-technical-shortcuts.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "increased-technical-shortcuts.md",
      "target": "time-pressure.md"
    },
    {
      "source": "increased-time-to-market.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "increased-time-to-market.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "increased-time-to-market.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "increased-time-to-market.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "increased-time-to-market.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "increased-time-to-market.md",
      "target": "long-release-cycles.md"
    },
    {
      "source": "increased-time-to-market.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "increasing-brittleness.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "increasing-brittleness.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "increasing-brittleness.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "increasing-brittleness.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "increasing-brittleness.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "increasing-brittleness.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "increasing-brittleness.md",
      "target": "lower-code-quality.md"
    },
    {
      "source": "increasing-brittleness.md",
      "target": "maintenance-cost-increase.md"
    },
    {
      "source": "increasing-brittleness.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "increasing-brittleness.md",
      "target": "procrastination-on-complex-tasks.md"
    },
    {
      "source": "increasing-brittleness.md",
      "target": "quality-blind-spots.md"
    },
    {
      "source": "increasing-brittleness.md",
      "target": "quality-compromises.md"
    },
    {
      "source": "increasing-brittleness.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "increasing-brittleness.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "increasing-brittleness.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "increasing-brittleness.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "individual-recognition-culture.md",
      "target": "blame-culture.md"
    },
    {
      "source": "individual-recognition-culture.md",
      "target": "clever-code.md"
    },
    {
      "source": "individual-recognition-culture.md",
      "target": "cv-driven-development.md"
    },
    {
      "source": "individual-recognition-culture.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "individual-recognition-culture.md",
      "target": "poor-communication.md"
    },
    {
      "source": "individual-recognition-culture.md",
      "target": "poor-teamwork.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "endianness-conversion-overhead.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "excessive-object-allocation.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "garbage-collection-pressure.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "high-client-side-resource-consumption.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "inefficient-frontend-code.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "inexperienced-developers.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "unbounded-data-structures.md"
    },
    {
      "source": "inefficient-code.md",
      "target": "unoptimized-file-access.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "database-query-performance-issues.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "incorrect-index-type.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "long-running-database-transactions.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "queries-that-prevent-index-usage.md"
    },
    {
      "source": "inefficient-database-indexing.md",
      "target": "slow-database-queries.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "difficult-developer-onboarding.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "reduced-individual-productivity.md"
    },
    {
      "source": "inefficient-development-environment.md",
      "target": "slow-development-velocity.md"
    },
    {
      "source": "inefficient-frontend-code.md",
      "target": "high-client-side-resource-consumption.md"
    },
    {
      "source": "inefficient-frontend-code.md",
      "target": "high-resource-utilization-on-client.md"
    },
    {
      "source": "inefficient-frontend-code.md",
      "target": "inefficient-code.md"
    },
    {
      "source": "inefficient-frontend-code.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "inefficient-processes.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "inefficient-processes.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "inefficient-processes.md",
      "target": "feedback-isolation.md"
    },
    {
      "source": "inefficient-processes.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "inefficient-processes.md",
      "target": "increased-manual-work.md"
    },
    {
      "source": "inefficient-processes.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "inefficient-processes.md",
      "target": "no-formal-change-control-process.md"
    },
    {
      "source": "inefficient-processes.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "inefficient-processes.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "inefficient-processes.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "inefficient-processes.md",
      "target": "unproductive-meetings.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "alignment-and-padding-issues.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "authentication-bypass-vulnerabilities.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "authorization-flaws.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "buffer-overflow-vulnerabilities.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "circular-references.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "clever-code.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "cross-site-scripting-vulnerabilities.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "data-structure-cache-inefficiency.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "database-query-performance-issues.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "error-message-information-disclosure.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "excessive-logging.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "excessive-object-allocation.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "false-sharing.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "global-state-and-side-effects.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "hardcoded-values.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "high-coupling-low-cohesion.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "improper-event-listener-management.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "inadequate-error-handling.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "inefficient-code.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "inefficient-database-indexing.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "inefficient-frontend-code.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "insufficient-code-review.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "log-injection-vulnerabilities.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "logging-configuration-issues.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "lower-code-quality.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "password-security-weaknesses.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "poor-caching-strategy.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "rest-api-design-issues.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "reviewer-inexperience.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "secret-management-problems.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "session-management-issues.md"
    },
    {
      "source": "inexperienced-developers.md",
      "target": "tangled-cross-cutting-concerns.md"
    },
    {
      "source": "information-decay.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "information-decay.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "information-decay.md",
      "target": "difficult-developer-onboarding.md"
    },
    {
      "source": "information-decay.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "information-decay.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "information-decay.md",
      "target": "hidden-dependencies.md"
    },
    {
      "source": "information-decay.md",
      "target": "information-fragmentation.md"
    },
    {
      "source": "information-decay.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "information-decay.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "information-decay.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "information-decay.md",
      "target": "legacy-system-documentation-archaeology.md"
    },
    {
      "source": "information-decay.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "information-decay.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "information-decay.md",
      "target": "unclear-documentation-ownership.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "duplicated-research-effort.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "information-fragmentation.md",
      "target": "unclear-documentation-ownership.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "data-protection-risk.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "log-injection-vulnerabilities.md"
    },
    {
      "source": "insufficient-audit-logging.md",
      "target": "logging-configuration-issues.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "high-defect-rate-in-production.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "improper-event-listener-management.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "inefficient-code.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "inefficient-frontend-code.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "n-plus-one-query-problem.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "queries-that-prevent-index-usage.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "sql-injection-vulnerabilities.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "stack-overflow-errors.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "team-members-not-engaged-in-review-process.md"
    },
    {
      "source": "insufficient-code-review.md",
      "target": "unreleased-resources.md"
    },
    {
      "source": "insufficient-design-skills.md",
      "target": "database-schema-design-problems.md"
    },
    {
      "source": "insufficient-design-skills.md",
      "target": "implementation-starts-without-design.md"
    },
    {
      "source": "insufficient-design-skills.md",
      "target": "insecure-data-transmission.md"
    },
    {
      "source": "insufficient-design-skills.md",
      "target": "insufficient-audit-logging.md"
    },
    {
      "source": "insufficient-design-skills.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "insufficient-design-skills.md",
      "target": "integer-overflow-underflow.md"
    },
    {
      "source": "insufficient-design-skills.md",
      "target": "memory-barrier-inefficiency.md"
    },
    {
      "source": "insufficient-design-skills.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "insufficient-design-skills.md",
      "target": "monolithic-architecture-constraints.md"
    },
    {
      "source": "insufficient-design-skills.md",
      "target": "monolithic-functions-and-classes.md"
    },
    {
      "source": "insufficient-design-skills.md",
      "target": "nitpicking-culture.md"
    },
    {
      "source": "insufficient-design-skills.md",
      "target": "poor-domain-model.md"
    },
    {
      "source": "insufficient-design-skills.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "insufficient-design-skills.md",
      "target": "poorly-defined-responsibilities.md"
    },
    {
      "source": "insufficient-design-skills.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "insufficient-design-skills.md",
      "target": "procedural-programming-in-oop-languages.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "authentication-bypass-vulnerabilities.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "authorization-flaws.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "buffer-overflow-vulnerabilities.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "cache-invalidation-problems.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "error-message-information-disclosure.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "high-defect-rate-in-production.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "inadequate-test-infrastructure.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "increased-risk-of-bugs.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "integer-overflow-underflow.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "quality-blind-spots.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "race-conditions.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "session-management-issues.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "test-debt.md"
    },
    {
      "source": "insufficient-testing.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "insufficient-worker-capacity.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "insufficient-worker-capacity.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "insufficient-worker-capacity.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "abi-compatibility-issues.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "dependency-version-conflicts.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "legacy-api-versioning-nightmare.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "obsolete-technologies.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "integration-difficulties.md",
      "target": "technology-stack-fragmentation.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "invisible-nature-of-technical-debt.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "knowledge-dependency.md",
      "target": "dependency-on-supplier.md"
    },
    {
      "source": "knowledge-dependency.md",
      "target": "inappropriate-skillset.md"
    },
    {
      "source": "knowledge-dependency.md",
      "target": "inconsistent-knowledge-acquisition.md"
    },
    {
      "source": "knowledge-dependency.md",
      "target": "inconsistent-onboarding-experience.md"
    },
    {
      "source": "knowledge-dependency.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "knowledge-dependency.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "knowledge-dependency.md",
      "target": "knowledge-silos.md"
    },
    {
      "source": "knowledge-dependency.md",
      "target": "mentor-burnout.md"
    },
    {
      "source": "knowledge-dependency.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "knowledge-dependency.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "complex-domain-model.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "cv-driven-development.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "data-protection-risk.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "decision-avoidance.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "dependency-on-supplier.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "information-decay.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "legacy-system-documentation-archaeology.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "n-plus-one-query-problem.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "reviewer-inexperience.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "stakeholder-developer-communication-gap.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "knowledge-gaps.md",
      "target": "superficial-code-reviews.md"
    },
    {
      "source": "knowledge-sharing-breakdown.md",
      "target": "duplicated-research-effort.md"
    },
    {
      "source": "knowledge-sharing-breakdown.md",
      "target": "inconsistent-knowledge-acquisition.md"
    },
    {
      "source": "knowledge-sharing-breakdown.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "knowledge-sharing-breakdown.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "knowledge-sharing-breakdown.md",
      "target": "knowledge-silos.md"
    },
    {
      "source": "knowledge-sharing-breakdown.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "knowledge-sharing-breakdown.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "blame-culture.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "capacity-mismatch.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "duplicated-research-effort.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "high-turnover.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "inconsistent-knowledge-acquisition.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "individual-recognition-culture.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "information-decay.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "information-fragmentation.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "legacy-skill-shortage.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "new-hire-frustration.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "poor-communication.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "poor-teamwork.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "reduced-review-participation.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "reduced-team-flexibility.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "reduced-team-productivity.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "resource-waste.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "review-process-avoidance.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "review-process-breakdown.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "skill-development-gaps.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "system-integration-blindness.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "team-churn-impact.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "team-members-not-engaged-in-review-process.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "team-silos.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "unclear-documentation-ownership.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "knowledge-silos.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "delayed-bug-fixes.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "inconsistent-codebase.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "inconsistent-coding-standards.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "inconsistent-onboarding-experience.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "insufficient-audit-logging.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "unclear-documentation-ownership.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "lack-of-ownership-and-accountability.md",
      "target": "undefined-code-style-guidelines.md"
    },
    {
      "source": "language-barriers.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "language-barriers.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "large-estimates-for-small-changes.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "large-estimates-for-small-changes.md",
      "target": "complex-implementation-paths.md"
    },
    {
      "source": "large-estimates-for-small-changes.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "large-estimates-for-small-changes.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "large-estimates-for-small-changes.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "large-estimates-for-small-changes.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "large-estimates-for-small-changes.md",
      "target": "legacy-business-logic-extraction-difficulty.md"
    },
    {
      "source": "large-estimates-for-small-changes.md",
      "target": "legacy-code-without-tests.md"
    },
    {
      "source": "large-estimates-for-small-changes.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "large-estimates-for-small-changes.md",
      "target": "poor-domain-model.md"
    },
    {
      "source": "large-feature-scope.md",
      "target": "large-pull-requests.md"
    },
    {
      "source": "large-feature-scope.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "large-feature-scope.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "author-frustration.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "extended-review-cycles.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "inadequate-code-reviews.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "inadequate-initial-reviews.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "insufficient-code-review.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "large-feature-scope.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "reduced-code-submission-frequency.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "review-process-breakdown.md"
    },
    {
      "source": "large-pull-requests.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "large-risky-releases.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "large-risky-releases.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "large-risky-releases.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "large-risky-releases.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "large-risky-releases.md",
      "target": "high-defect-rate-in-production.md"
    },
    {
      "source": "large-risky-releases.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "large-risky-releases.md",
      "target": "large-feature-scope.md"
    },
    {
      "source": "large-risky-releases.md",
      "target": "long-release-cycles.md"
    },
    {
      "source": "large-risky-releases.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "large-risky-releases.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "large-risky-releases.md",
      "target": "release-instability.md"
    },
    {
      "source": "large-risky-releases.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "lazy-loading.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "lazy-loading.md",
      "target": "n-plus-one-query-problem.md"
    },
    {
      "source": "lazy-loading.md",
      "target": "slow-database-queries.md"
    },
    {
      "source": "lazy-loading.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "dependency-version-conflicts.md"
    },
    {
      "source": "legacy-api-versioning-nightmare.md",
      "target": "rest-api-design-issues.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "data-migration-integrity-issues.md"
    },
    {
      "source": "legacy-business-logic-extraction-difficulty.md",
      "target": "legacy-system-documentation-archaeology.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "cross-site-scripting-vulnerabilities.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "developer-frustration-and-burnout.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "difficult-to-test-code.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "inadequate-test-infrastructure.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "incorrect-index-type.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "increased-manual-work.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "null-pointer-dereferences.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "poor-test-coverage.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "schema-evolution-paralysis.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "session-management-issues.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "sql-injection-vulnerabilities.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "unoptimized-file-access.md"
    },
    {
      "source": "legacy-code-without-tests.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "legacy-configuration-management-chaos.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "legacy-configuration-management-chaos.md",
      "target": "load-balancing-problems.md"
    },
    {
      "source": "legacy-configuration-management-chaos.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "legacy-configuration-management-chaos.md",
      "target": "secret-management-problems.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "obsolete-technologies.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "skill-development-gaps.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "technology-isolation.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "technology-stack-fragmentation.md"
    },
    {
      "source": "legacy-skill-shortage.md",
      "target": "vendor-dependency-entrapment.md"
    },
    {
      "source": "legacy-system-documentation-archaeology.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "legacy-system-documentation-archaeology.md",
      "target": "information-decay.md"
    },
    {
      "source": "legal-disputes.md",
      "target": "authentication-bypass-vulnerabilities.md"
    },
    {
      "source": "legal-disputes.md",
      "target": "authorization-flaws.md"
    },
    {
      "source": "legal-disputes.md",
      "target": "cross-site-scripting-vulnerabilities.md"
    },
    {
      "source": "legal-disputes.md",
      "target": "poor-contract-design.md"
    },
    {
      "source": "limited-team-learning.md",
      "target": "inadequate-code-reviews.md"
    },
    {
      "source": "load-balancing-problems.md",
      "target": "rate-limiting-issues.md"
    },
    {
      "source": "lock-contention.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "lock-contention.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "lock-contention.md",
      "target": "long-running-database-transactions.md"
    },
    {
      "source": "lock-contention.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "log-injection-vulnerabilities.md",
      "target": "logging-configuration-issues.md"
    },
    {
      "source": "log-spam.md",
      "target": "logging-configuration-issues.md"
    },
    {
      "source": "logging-configuration-issues.md",
      "target": "error-message-information-disclosure.md"
    },
    {
      "source": "logging-configuration-issues.md",
      "target": "excessive-logging.md"
    },
    {
      "source": "logging-configuration-issues.md",
      "target": "log-injection-vulnerabilities.md"
    },
    {
      "source": "logging-configuration-issues.md",
      "target": "log-spam.md"
    },
    {
      "source": "long-build-and-test-times.md",
      "target": "circular-dependency-problems.md"
    },
    {
      "source": "long-build-and-test-times.md",
      "target": "dependency-version-conflicts.md"
    },
    {
      "source": "long-build-and-test-times.md",
      "target": "flaky-tests.md"
    },
    {
      "source": "long-build-and-test-times.md",
      "target": "inadequate-test-infrastructure.md"
    },
    {
      "source": "long-build-and-test-times.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "long-build-and-test-times.md",
      "target": "long-release-cycles.md"
    },
    {
      "source": "long-build-and-test-times.md",
      "target": "monolithic-architecture-constraints.md"
    },
    {
      "source": "long-lived-feature-branches.md",
      "target": "large-feature-scope.md"
    },
    {
      "source": "long-lived-feature-branches.md",
      "target": "large-pull-requests.md"
    },
    {
      "source": "long-lived-feature-branches.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "long-lived-feature-branches.md",
      "target": "merge-conflicts.md"
    },
    {
      "source": "long-release-cycles.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "long-release-cycles.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "long-release-cycles.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "long-release-cycles.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "long-release-cycles.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "long-release-cycles.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "long-release-cycles.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "long-release-cycles.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "long-release-cycles.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "long-running-transactions.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "deadline-pressure.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "developer-frustration-and-burnout.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "development-disruption.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "fear-of-conflict.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "inadequate-code-reviews.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "inappropriate-skillset.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "increased-risk-of-bugs.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "inefficient-code.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "inefficient-frontend-code.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "inexperienced-developers.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "insufficient-code-review.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "mental-fatigue.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "quality-compromises.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "team-churn-impact.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "team-demoralization.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "team-members-not-engaged-in-review-process.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "time-pressure.md"
    },
    {
      "source": "lower-code-quality.md",
      "target": "unmotivated-employees.md"
    },
    {
      "source": "maintenance-bottlenecks.md",
      "target": "clever-code.md"
    },
    {
      "source": "maintenance-bottlenecks.md",
      "target": "legacy-skill-shortage.md"
    },
    {
      "source": "maintenance-bottlenecks.md",
      "target": "mentor-burnout.md"
    },
    {
      "source": "maintenance-bottlenecks.md",
      "target": "monolithic-architecture-constraints.md"
    },
    {
      "source": "maintenance-bottlenecks.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "maintenance-bottlenecks.md",
      "target": "single-entry-point-design.md"
    },
    {
      "source": "maintenance-cost-increase.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "maintenance-cost-increase.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "maintenance-cost-increase.md",
      "target": "increased-cost-of-development.md"
    },
    {
      "source": "maintenance-cost-increase.md",
      "target": "increased-customer-support-load.md"
    },
    {
      "source": "maintenance-cost-increase.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "maintenance-cost-increase.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "maintenance-cost-increase.md",
      "target": "regulatory-compliance-drift.md"
    },
    {
      "source": "maintenance-cost-increase.md",
      "target": "vendor-dependency-entrapment.md"
    },
    {
      "source": "maintenance-cost-increase.md",
      "target": "vendor-dependency.md"
    },
    {
      "source": "maintenance-cost-increase.md",
      "target": "vendor-lock-in.md"
    },
    {
      "source": "maintenance-cost-increase.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "maintenance-cost-increase.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "maintenance-overhead.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "maintenance-overhead.md",
      "target": "code-duplication.md"
    },
    {
      "source": "maintenance-overhead.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "maintenance-overhead.md",
      "target": "lower-code-quality.md"
    },
    {
      "source": "maintenance-overhead.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "maintenance-overhead.md",
      "target": "second-system-effect.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "inability-to-innovate.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "invisible-nature-of-technical-debt.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "legacy-code-without-tests.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "maintenance-paralysis.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "inadequate-configuration-management.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "increased-manual-work.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "long-release-cycles.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "manual-deployment-processes.md",
      "target": "release-instability.md"
    },
    {
      "source": "market-pressure.md",
      "target": "competing-priorities.md"
    },
    {
      "source": "market-pressure.md",
      "target": "cv-driven-development.md"
    },
    {
      "source": "market-pressure.md",
      "target": "deadline-pressure.md"
    },
    {
      "source": "market-pressure.md",
      "target": "feature-factory.md"
    },
    {
      "source": "market-pressure.md",
      "target": "frequent-changes-to-requirements.md"
    },
    {
      "source": "market-pressure.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "market-pressure.md",
      "target": "poor-planning.md"
    },
    {
      "source": "market-pressure.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "market-pressure.md",
      "target": "time-pressure.md"
    },
    {
      "source": "market-pressure.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "memory-fragmentation.md",
      "target": "alignment-and-padding-issues.md"
    },
    {
      "source": "memory-fragmentation.md",
      "target": "excessive-object-allocation.md"
    },
    {
      "source": "memory-fragmentation.md",
      "target": "memory-leaks.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "circular-references.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "garbage-collection-pressure.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "high-client-side-resource-consumption.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "high-resource-utilization-on-client.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "improper-event-listener-management.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "memory-fragmentation.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "system-outages.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "unbounded-data-structures.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "unreleased-resources.md"
    },
    {
      "source": "memory-leaks.md",
      "target": "virtual-memory-thrashing.md"
    },
    {
      "source": "memory-swapping.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "memory-swapping.md",
      "target": "memory-leaks.md"
    },
    {
      "source": "memory-swapping.md",
      "target": "resource-contention.md"
    },
    {
      "source": "mental-fatigue.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "mental-fatigue.md",
      "target": "context-switching-overhead.md"
    },
    {
      "source": "mental-fatigue.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "difficult-developer-onboarding.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "high-turnover.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "inadequate-mentoring-structure.md"
    },
    {
      "source": "mentor-burnout.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "bloated-class.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "large-feature-scope.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "monolithic-architecture-constraints.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "monolithic-functions-and-classes.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "merge-conflicts.md",
      "target": "team-coordination-issues.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "decision-avoidance.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "developer-frustration-and-burnout.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "fear-of-failure.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "individual-recognition-culture.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "poor-teamwork.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "stakeholder-dissatisfaction.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "stakeholder-frustration.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "unmotivated-employees.md"
    },
    {
      "source": "micromanagement-culture.md",
      "target": "work-blocking.md"
    },
    {
      "source": "microservice-communication-overhead.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "microservice-communication-overhead.md",
      "target": "network-latency.md"
    },
    {
      "source": "misaligned-deliverables.md",
      "target": "assumption-based-development.md"
    },
    {
      "source": "misaligned-deliverables.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "misaligned-deliverables.md",
      "target": "communication-risk-outside-project.md"
    },
    {
      "source": "misaligned-deliverables.md",
      "target": "feedback-isolation.md"
    },
    {
      "source": "misaligned-deliverables.md",
      "target": "inadequate-requirements-gathering.md"
    },
    {
      "source": "misaligned-deliverables.md",
      "target": "language-barriers.md"
    },
    {
      "source": "misaligned-deliverables.md",
      "target": "no-continuous-feedback-loop.md"
    },
    {
      "source": "misaligned-deliverables.md",
      "target": "poor-contract-design.md"
    },
    {
      "source": "misaligned-deliverables.md",
      "target": "product-direction-chaos.md"
    },
    {
      "source": "misaligned-deliverables.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "misaligned-deliverables.md",
      "target": "scope-change-resistance.md"
    },
    {
      "source": "misaligned-deliverables.md",
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "misaligned-deliverables.md",
      "target": "stakeholder-developer-communication-gap.md"
    },
    {
      "source": "misaligned-deliverables.md",
      "target": "stakeholder-dissatisfaction.md"
    },
    {
      "source": "misaligned-deliverables.md",
      "target": "team-dysfunction.md"
    },
    {
      "source": "misconfigured-connection-pools.md",
      "target": "high-connection-count.md"
    },
    {
      "source": "misconfigured-connection-pools.md",
      "target": "incorrect-max-connection-pool-size.md"
    },
    {
      "source": "misconfigured-connection-pools.md",
      "target": "upstream-timeouts.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "analysis-paralysis.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "cascade-delays.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "development-disruption.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "frequent-changes-to-requirements.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "gold-plating.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "market-pressure.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "no-formal-change-control-process.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "planning-credibility-issues.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "poor-planning.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "slow-development-velocity.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "team-confusion.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "missed-deadlines.md",
      "target": "unrealistic-schedule.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "quality-blind-spots.md"
    },
    {
      "source": "missing-end-to-end-tests.md",
      "target": "system-integration-blindness.md"
    },
    {
      "source": "missing-rollback-strategy.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "missing-rollback-strategy.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "missing-rollback-strategy.md",
      "target": "long-release-cycles.md"
    },
    {
      "source": "misunderstanding-of-oop.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "misunderstanding-of-oop.md",
      "target": "excessive-object-allocation.md"
    },
    {
      "source": "misunderstanding-of-oop.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "misunderstanding-of-oop.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "misunderstanding-of-oop.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "misunderstanding-of-oop.md",
      "target": "procedural-background.md"
    },
    {
      "source": "misunderstanding-of-oop.md",
      "target": "procedural-programming-in-oop-languages.md"
    },
    {
      "source": "mixed-coding-styles.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "mixed-coding-styles.md",
      "target": "procedural-programming-in-oop-languages.md"
    },
    {
      "source": "mixed-coding-styles.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "mixed-coding-styles.md",
      "target": "undefined-code-style-guidelines.md"
    },
    {
      "source": "modernization-roi-justification-failure.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "modernization-roi-justification-failure.md",
      "target": "legacy-business-logic-extraction-difficulty.md"
    },
    {
      "source": "modernization-roi-justification-failure.md",
      "target": "legacy-system-documentation-archaeology.md"
    },
    {
      "source": "modernization-roi-justification-failure.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "modernization-roi-justification-failure.md",
      "target": "obsolete-technologies.md"
    },
    {
      "source": "modernization-strategy-paralysis.md",
      "target": "cross-system-data-synchronization-problems.md"
    },
    {
      "source": "modernization-strategy-paralysis.md",
      "target": "legacy-business-logic-extraction-difficulty.md"
    },
    {
      "source": "modernization-strategy-paralysis.md",
      "target": "legacy-skill-shortage.md"
    },
    {
      "source": "modernization-strategy-paralysis.md",
      "target": "modernization-roi-justification-failure.md"
    },
    {
      "source": "monitoring-gaps.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "monitoring-gaps.md",
      "target": "insufficient-audit-logging.md"
    },
    {
      "source": "monitoring-gaps.md",
      "target": "load-balancing-problems.md"
    },
    {
      "source": "monitoring-gaps.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "monitoring-gaps.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "monitoring-gaps.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "monitoring-gaps.md",
      "target": "system-outages.md"
    },
    {
      "source": "monitoring-gaps.md",
      "target": "unused-indexes.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "complex-implementation-paths.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "dependency-version-conflicts.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "inefficient-development-environment.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "large-feature-scope.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "scaling-inefficiencies.md"
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
      "target": "single-entry-point-design.md"
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
      "target": "technical-architecture-limitations.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "monolithic-architecture-constraints.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "circular-dependency-problems.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "difficult-to-test-code.md"
    },
    {
      "source": "monolithic-functions-and-classes.md",
      "target": "merge-conflicts.md"
    },
    {
      "source": "n-plus-one-query-problem.md",
      "target": "database-query-performance-issues.md"
    },
    {
      "source": "n-plus-one-query-problem.md",
      "target": "graphql-complexity-issues.md"
    },
    {
      "source": "n-plus-one-query-problem.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "n-plus-one-query-problem.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "n-plus-one-query-problem.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "n-plus-one-query-problem.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "n-plus-one-query-problem.md",
      "target": "log-spam.md"
    },
    {
      "source": "n-plus-one-query-problem.md",
      "target": "resource-contention.md"
    },
    {
      "source": "negative-brand-perception.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "negative-brand-perception.md",
      "target": "cross-site-scripting-vulnerabilities.md"
    },
    {
      "source": "negative-brand-perception.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "negative-brand-perception.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "negative-brand-perception.md",
      "target": "negative-user-feedback.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "assumption-based-development.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "database-query-performance-issues.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "feature-gaps.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "high-client-side-resource-consumption.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "inadequate-onboarding.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "increased-error-rates.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "n-plus-one-query-problem.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "negative-brand-perception.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "network-latency.md"
    },
    {
      "source": "negative-user-feedback.md",
      "target": "poor-user-experience-ux-design.md"
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
      "source": "negative-user-feedback.md",
      "target": "user-trust-erosion.md"
    },
    {
      "source": "network-latency.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "network-latency.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "network-latency.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "network-latency.md",
      "target": "service-timeouts.md"
    },
    {
      "source": "network-latency.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "difficult-developer-onboarding.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "inconsistent-onboarding-experience.md"
    },
    {
      "source": "new-hire-frustration.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "nitpicking-culture.md",
      "target": "author-frustration.md"
    },
    {
      "source": "nitpicking-culture.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "nitpicking-culture.md",
      "target": "defensive-coding-practices.md"
    },
    {
      "source": "nitpicking-culture.md",
      "target": "inadequate-initial-reviews.md"
    },
    {
      "source": "nitpicking-culture.md",
      "target": "perfectionist-review-culture.md"
    },
    {
      "source": "nitpicking-culture.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "nitpicking-culture.md",
      "target": "superficial-code-reviews.md"
    },
    {
      "source": "no-continuous-feedback-loop.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "no-continuous-feedback-loop.md",
      "target": "reduced-code-submission-frequency.md"
    },
    {
      "source": "no-formal-change-control-process.md",
      "target": "changing-project-scope.md"
    },
    {
      "source": "no-formal-change-control-process.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "no-formal-change-control-process.md",
      "target": "feature-creep.md"
    },
    {
      "source": "no-formal-change-control-process.md",
      "target": "gold-plating.md"
    },
    {
      "source": "no-formal-change-control-process.md",
      "target": "scope-creep.md"
    },
    {
      "source": "obsolete-technologies.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "obsolete-technologies.md",
      "target": "data-migration-integrity-issues.md"
    },
    {
      "source": "obsolete-technologies.md",
      "target": "inadequate-test-infrastructure.md"
    },
    {
      "source": "obsolete-technologies.md",
      "target": "insecure-data-transmission.md"
    },
    {
      "source": "obsolete-technologies.md",
      "target": "integration-difficulties.md"
    },
    {
      "source": "obsolete-technologies.md",
      "target": "legacy-api-versioning-nightmare.md"
    },
    {
      "source": "obsolete-technologies.md",
      "target": "legacy-configuration-management-chaos.md"
    },
    {
      "source": "obsolete-technologies.md",
      "target": "legacy-skill-shortage.md"
    },
    {
      "source": "obsolete-technologies.md",
      "target": "maintenance-cost-increase.md"
    },
    {
      "source": "obsolete-technologies.md",
      "target": "modernization-roi-justification-failure.md"
    },
    {
      "source": "obsolete-technologies.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "obsolete-technologies.md",
      "target": "password-security-weaknesses.md"
    },
    {
      "source": "obsolete-technologies.md",
      "target": "serialization-deserialization-bottlenecks.md"
    },
    {
      "source": "obsolete-technologies.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "obsolete-technologies.md",
      "target": "technology-isolation.md"
    },
    {
      "source": "obsolete-technologies.md",
      "target": "technology-stack-fragmentation.md"
    },
    {
      "source": "operational-overhead.md",
      "target": "increased-customer-support-load.md"
    },
    {
      "source": "operational-overhead.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "organizational-structure-mismatch.md",
      "target": "capacity-mismatch.md"
    },
    {
      "source": "organizational-structure-mismatch.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "organizational-structure-mismatch.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "organizational-structure-mismatch.md",
      "target": "team-silos.md"
    },
    {
      "source": "over-reliance-on-utility-classes.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "over-reliance-on-utility-classes.md",
      "target": "procedural-background.md"
    },
    {
      "source": "overworked-teams.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "overworked-teams.md",
      "target": "inadequate-code-reviews.md"
    },
    {
      "source": "overworked-teams.md",
      "target": "inadequate-mentoring-structure.md"
    },
    {
      "source": "overworked-teams.md",
      "target": "inconsistent-onboarding-experience.md"
    },
    {
      "source": "overworked-teams.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "overworked-teams.md",
      "target": "insufficient-code-review.md"
    },
    {
      "source": "overworked-teams.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "overworked-teams.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "overworked-teams.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "partial-bug-fixes.md",
      "target": "code-duplication.md"
    },
    {
      "source": "partial-bug-fixes.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "partial-bug-fixes.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "partial-bug-fixes.md",
      "target": "regression-bugs.md"
    },
    {
      "source": "partial-bug-fixes.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "past-negative-experiences.md",
      "target": "fear-of-failure.md"
    },
    {
      "source": "past-negative-experiences.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "past-negative-experiences.md",
      "target": "second-system-effect.md"
    },
    {
      "source": "perfectionist-culture.md",
      "target": "analysis-paralysis.md"
    },
    {
      "source": "perfectionist-culture.md",
      "target": "perfectionist-review-culture.md"
    },
    {
      "source": "perfectionist-review-culture.md",
      "target": "defensive-coding-practices.md"
    },
    {
      "source": "perfectionist-review-culture.md",
      "target": "extended-review-cycles.md"
    },
    {
      "source": "perfectionist-review-culture.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "planning-credibility-issues.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "planning-credibility-issues.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "planning-credibility-issues.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "planning-credibility-issues.md",
      "target": "poor-planning.md"
    },
    {
      "source": "planning-credibility-issues.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "planning-credibility-issues.md",
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "planning-dysfunction.md",
      "target": "planning-credibility-issues.md"
    },
    {
      "source": "planning-dysfunction.md",
      "target": "unrealistic-schedule.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "cache-invalidation-problems.md"
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
      "target": "external-service-delays.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "graphql-complexity-issues.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "unbounded-data-growth.md"
    },
    {
      "source": "poor-caching-strategy.md",
      "target": "unbounded-data-structures.md"
    },
    {
      "source": "poor-communication.md",
      "target": "individual-recognition-culture.md"
    },
    {
      "source": "poor-communication.md",
      "target": "poor-project-control.md"
    },
    {
      "source": "poor-communication.md",
      "target": "team-confusion.md"
    },
    {
      "source": "poor-communication.md",
      "target": "team-dysfunction.md"
    },
    {
      "source": "poor-communication.md",
      "target": "team-silos.md"
    },
    {
      "source": "poor-communication.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "poor-contract-design.md",
      "target": "dependency-on-supplier.md"
    },
    {
      "source": "poor-contract-design.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "poor-contract-design.md",
      "target": "scope-change-resistance.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "cache-invalidation-problems.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "data-migration-integrity-issues.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "high-turnover.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "inadequate-configuration-management.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "inefficient-database-indexing.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "information-decay.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "information-fragmentation.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "legacy-business-logic-extraction-difficulty.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "legacy-configuration-management-chaos.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "new-hire-frustration.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "rate-limiting-issues.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "slow-development-velocity.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "team-churn-impact.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "unused-indexes.md"
    },
    {
      "source": "poor-documentation.md",
      "target": "user-confusion.md"
    },
    {
      "source": "poor-domain-model.md",
      "target": "complex-domain-model.md"
    },
    {
      "source": "poor-encapsulation.md",
      "target": "circular-references.md"
    },
    {
      "source": "poor-encapsulation.md",
      "target": "global-state-and-side-effects.md"
    },
    {
      "source": "poor-encapsulation.md",
      "target": "hidden-dependencies.md"
    },
    {
      "source": "poor-encapsulation.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "poor-encapsulation.md",
      "target": "high-coupling-low-cohesion.md"
    },
    {
      "source": "poor-encapsulation.md",
      "target": "procedural-background.md"
    },
    {
      "source": "poor-encapsulation.md",
      "target": "procedural-programming-in-oop-languages.md"
    },
    {
      "source": "poor-encapsulation.md",
      "target": "ripple-effect-of-changes.md"
    },
    {
      "source": "poor-interfaces-between-applications.md",
      "target": "abi-compatibility-issues.md"
    },
    {
      "source": "poor-interfaces-between-applications.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "poor-interfaces-between-applications.md",
      "target": "cross-system-data-synchronization-problems.md"
    },
    {
      "source": "poor-interfaces-between-applications.md",
      "target": "endianness-conversion-overhead.md"
    },
    {
      "source": "poor-interfaces-between-applications.md",
      "target": "legacy-api-versioning-nightmare.md"
    },
    {
      "source": "poor-interfaces-between-applications.md",
      "target": "rest-api-design-issues.md"
    },
    {
      "source": "poor-naming-conventions.md",
      "target": "complex-and-obscure-logic.md"
    },
    {
      "source": "poor-naming-conventions.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "poor-naming-conventions.md",
      "target": "difficult-to-understand-code.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "poor-operational-concept.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "poor-planning.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "poor-planning.md",
      "target": "capacity-mismatch.md"
    },
    {
      "source": "poor-planning.md",
      "target": "cascade-delays.md"
    },
    {
      "source": "poor-planning.md",
      "target": "competing-priorities.md"
    },
    {
      "source": "poor-planning.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "poor-planning.md",
      "target": "deadline-pressure.md"
    },
    {
      "source": "poor-planning.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "poor-planning.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "poor-planning.md",
      "target": "duplicated-work.md"
    },
    {
      "source": "poor-planning.md",
      "target": "feedback-isolation.md"
    },
    {
      "source": "poor-planning.md",
      "target": "hardcoded-values.md"
    },
    {
      "source": "poor-planning.md",
      "target": "inadequate-test-data-management.md"
    },
    {
      "source": "poor-planning.md",
      "target": "inappropriate-skillset.md"
    },
    {
      "source": "poor-planning.md",
      "target": "insufficient-worker-capacity.md"
    },
    {
      "source": "poor-planning.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "poor-planning.md",
      "target": "planning-credibility-issues.md"
    },
    {
      "source": "poor-planning.md",
      "target": "poor-contract-design.md"
    },
    {
      "source": "poor-planning.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "poor-planning.md",
      "target": "poor-project-control.md"
    },
    {
      "source": "poor-planning.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "poor-planning.md",
      "target": "project-authority-vacuum.md"
    },
    {
      "source": "poor-planning.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "poor-planning.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "poor-planning.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "poor-planning.md",
      "target": "rapid-team-growth.md"
    },
    {
      "source": "poor-planning.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "poor-planning.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "poor-planning.md",
      "target": "unrealistic-schedule.md"
    },
    {
      "source": "poor-planning.md",
      "target": "vendor-dependency.md"
    },
    {
      "source": "poor-planning.md",
      "target": "vendor-lock-in.md"
    },
    {
      "source": "poor-planning.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "communication-risk-outside-project.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "no-formal-change-control-process.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "poor-planning.md"
    },
    {
      "source": "poor-project-control.md",
      "target": "scope-creep.md"
    },
    {
      "source": "poor-system-environment.md",
      "target": "inefficient-development-environment.md"
    },
    {
      "source": "poor-system-environment.md",
      "target": "interrupt-overhead.md"
    },
    {
      "source": "poor-system-environment.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "rapid-team-growth.md"
    },
    {
      "source": "poor-teamwork.md",
      "target": "reduced-team-flexibility.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "difficult-to-test-code.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "long-release-cycles.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "null-pointer-dereferences.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "quality-blind-spots.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "race-conditions.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "release-instability.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "poor-test-coverage.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "poor-user-experience-ux-design.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "poor-user-experience-ux-design.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "poor-user-experience-ux-design.md",
      "target": "feature-bloat.md"
    },
    {
      "source": "poor-user-experience-ux-design.md",
      "target": "inadequate-onboarding.md"
    },
    {
      "source": "poor-user-experience-ux-design.md",
      "target": "increased-customer-support-load.md"
    },
    {
      "source": "poor-user-experience-ux-design.md",
      "target": "negative-brand-perception.md"
    },
    {
      "source": "poor-user-experience-ux-design.md",
      "target": "negative-user-feedback.md"
    },
    {
      "source": "poor-user-experience-ux-design.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "poor-user-experience-ux-design.md",
      "target": "user-confusion.md"
    },
    {
      "source": "poor-user-experience-ux-design.md",
      "target": "user-frustration.md"
    },
    {
      "source": "poorly-defined-responsibilities.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "poorly-defined-responsibilities.md",
      "target": "circular-dependency-problems.md"
    },
    {
      "source": "poorly-defined-responsibilities.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "poorly-defined-responsibilities.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "poorly-defined-responsibilities.md",
      "target": "high-coupling-low-cohesion.md"
    },
    {
      "source": "poorly-defined-responsibilities.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "poorly-defined-responsibilities.md",
      "target": "monolithic-functions-and-classes.md"
    },
    {
      "source": "poorly-defined-responsibilities.md",
      "target": "no-formal-change-control-process.md"
    },
    {
      "source": "poorly-defined-responsibilities.md",
      "target": "team-coordination-issues.md"
    },
    {
      "source": "poorly-defined-responsibilities.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "poorly-defined-responsibilities.md",
      "target": "unclear-documentation-ownership.md"
    },
    {
      "source": "poorly-defined-responsibilities.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "poorly-defined-responsibilities.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "poorly-defined-responsibilities.md",
      "target": "unproductive-meetings.md"
    },
    {
      "source": "power-struggles.md",
      "target": "competing-priorities.md"
    },
    {
      "source": "power-struggles.md",
      "target": "individual-recognition-culture.md"
    },
    {
      "source": "power-struggles.md",
      "target": "micromanagement-culture.md"
    },
    {
      "source": "power-struggles.md",
      "target": "priority-thrashing.md"
    },
    {
      "source": "power-struggles.md",
      "target": "product-direction-chaos.md"
    },
    {
      "source": "power-struggles.md",
      "target": "project-authority-vacuum.md"
    },
    {
      "source": "premature-technology-introduction.md",
      "target": "cv-driven-development.md"
    },
    {
      "source": "priority-thrashing.md",
      "target": "context-switching-overhead.md"
    },
    {
      "source": "priority-thrashing.md",
      "target": "development-disruption.md"
    },
    {
      "source": "priority-thrashing.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "priority-thrashing.md",
      "target": "power-struggles.md"
    },
    {
      "source": "priority-thrashing.md",
      "target": "product-direction-chaos.md"
    },
    {
      "source": "priority-thrashing.md",
      "target": "unclear-goals-and-priorities.md"
    },
    {
      "source": "procedural-background.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "procedural-background.md",
      "target": "over-reliance-on-utility-classes.md"
    },
    {
      "source": "procedural-background.md",
      "target": "procedural-programming-in-oop-languages.md"
    },
    {
      "source": "procedural-programming-in-oop-languages.md",
      "target": "global-state-and-side-effects.md"
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
      "source": "process-design-flaws.md",
      "target": "inadequate-configuration-management.md"
    },
    {
      "source": "process-design-flaws.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "process-design-flaws.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "procrastination-on-complex-tasks.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "changing-project-scope.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "feature-creep.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "frequent-changes-to-requirements.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "market-pressure.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "priority-thrashing.md"
    },
    {
      "source": "product-direction-chaos.md",
      "target": "unclear-goals-and-priorities.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "communication-risk-outside-project.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "power-struggles.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "product-direction-chaos.md"
    },
    {
      "source": "project-authority-vacuum.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "project-resource-constraints.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "project-resource-constraints.md",
      "target": "inadequate-test-infrastructure.md"
    },
    {
      "source": "project-resource-constraints.md",
      "target": "project-authority-vacuum.md"
    },
    {
      "source": "project-resource-constraints.md",
      "target": "quality-compromises.md"
    },
    {
      "source": "project-resource-constraints.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "project-resource-constraints.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "flaky-tests.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "inadequate-test-data-management.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "nitpicking-culture.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "poor-test-coverage.md"
    },
    {
      "source": "quality-blind-spots.md",
      "target": "quality-compromises.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "competing-priorities.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "cross-site-scripting-vulnerabilities.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "data-protection-risk.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "deadline-pressure.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "password-security-weaknesses.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "poor-contract-design.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "priority-thrashing.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "quality-blind-spots.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "reduced-feature-quality.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "scope-creep.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "time-pressure.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "unrealistic-schedule.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "vendor-dependency.md"
    },
    {
      "source": "quality-compromises.md",
      "target": "vendor-lock-in.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "quality-blind-spots.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "quality-compromises.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "reduced-feature-quality.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "test-debt.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "unrealistic-schedule.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "user-trust-erosion.md"
    },
    {
      "source": "quality-degradation.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "race-conditions.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "race-conditions.md",
      "target": "dma-coherency-issues.md"
    },
    {
      "source": "race-conditions.md",
      "target": "global-state-and-side-effects.md"
    },
    {
      "source": "race-conditions.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "rapid-prototyping-becoming-production.md",
      "target": "authentication-bypass-vulnerabilities.md"
    },
    {
      "source": "rapid-prototyping-becoming-production.md",
      "target": "legacy-code-without-tests.md"
    },
    {
      "source": "rapid-prototyping-becoming-production.md",
      "target": "poor-caching-strategy.md"
    },
    {
      "source": "rapid-system-changes.md",
      "target": "api-versioning-conflicts.md"
    },
    {
      "source": "rapid-system-changes.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "rapid-system-changes.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "rapid-team-growth.md",
      "target": "inadequate-mentoring-structure.md"
    },
    {
      "source": "rapid-team-growth.md",
      "target": "inappropriate-skillset.md"
    },
    {
      "source": "rapid-team-growth.md",
      "target": "language-barriers.md"
    },
    {
      "source": "rapid-team-growth.md",
      "target": "mentor-burnout.md"
    },
    {
      "source": "rapid-team-growth.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "rapid-team-growth.md",
      "target": "technology-stack-fragmentation.md"
    },
    {
      "source": "reduced-code-submission-frequency.md",
      "target": "author-frustration.md"
    },
    {
      "source": "reduced-code-submission-frequency.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "reduced-code-submission-frequency.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "reduced-code-submission-frequency.md",
      "target": "defensive-coding-practices.md"
    },
    {
      "source": "reduced-code-submission-frequency.md",
      "target": "extended-review-cycles.md"
    },
    {
      "source": "reduced-code-submission-frequency.md",
      "target": "large-pull-requests.md"
    },
    {
      "source": "reduced-code-submission-frequency.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "reduced-code-submission-frequency.md",
      "target": "perfectionist-review-culture.md"
    },
    {
      "source": "reduced-code-submission-frequency.md",
      "target": "reduced-review-participation.md"
    },
    {
      "source": "reduced-feature-quality.md",
      "target": "feature-factory.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "context-switching-overhead.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "increased-manual-work.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "mental-fatigue.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "mentor-burnout.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "micromanagement-culture.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "reduced-team-productivity.md"
    },
    {
      "source": "reduced-individual-productivity.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "blame-culture.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "developer-frustration-and-burnout.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "fear-of-failure.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "maintenance-cost-increase.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "perfectionist-culture.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "team-demoralization.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "technology-isolation.md"
    },
    {
      "source": "reduced-innovation.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "reduced-predictability.md",
      "target": "reduced-team-flexibility.md"
    },
    {
      "source": "reduced-review-participation.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "reduced-review-participation.md",
      "target": "review-process-breakdown.md"
    },
    {
      "source": "reduced-review-participation.md",
      "target": "reviewer-anxiety.md"
    },
    {
      "source": "reduced-review-participation.md",
      "target": "unmotivated-employees.md"
    },
    {
      "source": "reduced-team-flexibility.md",
      "target": "dependency-on-supplier.md"
    },
    {
      "source": "reduced-team-flexibility.md",
      "target": "knowledge-silos.md"
    },
    {
      "source": "reduced-team-flexibility.md",
      "target": "team-silos.md"
    },
    {
      "source": "reduced-team-flexibility.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "reduced-team-flexibility.md",
      "target": "vendor-dependency.md"
    },
    {
      "source": "reduced-team-flexibility.md",
      "target": "vendor-lock-in.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "competing-priorities.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "developer-frustration-and-burnout.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "development-disruption.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "difficult-developer-onboarding.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "duplicated-research-effort.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "duplicated-work.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "inefficient-development-environment.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "knowledge-silos.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "poor-teamwork.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "rapid-team-growth.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "reduced-individual-productivity.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "reduced-team-flexibility.md"
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
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "team-churn-impact.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "team-confusion.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "team-coordination-issues.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "team-demoralization.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "team-dysfunction.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "uneven-work-flow.md"
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
      "target": "wasted-development-effort.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "work-blocking.md"
    },
    {
      "source": "reduced-team-productivity.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "blame-culture.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "bloated-class.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "high-coupling-low-cohesion.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "poor-test-coverage.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "regression-bugs.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "regulatory-compliance-drift.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "refactoring-avoidance.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "bloated-class.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "difficult-to-test-code.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "global-state-and-side-effects.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "hidden-dependencies.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "inadequate-code-reviews.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "inadequate-integration-tests.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "insufficient-code-review.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "legacy-code-without-tests.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "lower-code-quality.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "negative-user-feedback.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "poor-domain-model.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "poor-test-coverage.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "rapid-system-changes.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "review-process-breakdown.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "ripple-effect-of-changes.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "superficial-code-reviews.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "synchronization-problems.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "test-debt.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "regression-bugs.md",
      "target": "unpredictable-system-behavior.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "authentication-bypass-vulnerabilities.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "authorization-flaws.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "data-protection-risk.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "insecure-data-transmission.md"
    },
    {
      "source": "regulatory-compliance-drift.md",
      "target": "insufficient-audit-logging.md"
    },
    {
      "source": "release-anxiety.md",
      "target": "complex-deployment-process.md"
    },
    {
      "source": "release-anxiety.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "release-anxiety.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "release-anxiety.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "release-anxiety.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "release-anxiety.md",
      "target": "long-release-cycles.md"
    },
    {
      "source": "release-anxiety.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "release-anxiety.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "release-anxiety.md",
      "target": "release-instability.md"
    },
    {
      "source": "release-instability.md",
      "target": "deployment-environment-inconsistencies.md"
    },
    {
      "source": "release-instability.md",
      "target": "inadequate-integration-tests.md"
    },
    {
      "source": "release-instability.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "release-instability.md",
      "target": "increased-bug-count.md"
    },
    {
      "source": "release-instability.md",
      "target": "increased-error-rates.md"
    },
    {
      "source": "release-instability.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "release-instability.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "release-instability.md",
      "target": "unreleased-resources.md"
    },
    {
      "source": "release-instability.md",
      "target": "user-frustration.md"
    },
    {
      "source": "release-instability.md",
      "target": "user-trust-erosion.md"
    },
    {
      "source": "requirements-ambiguity.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "requirements-ambiguity.md",
      "target": "complex-domain-model.md"
    },
    {
      "source": "requirements-ambiguity.md",
      "target": "implementation-rework.md"
    },
    {
      "source": "requirements-ambiguity.md",
      "target": "implementation-starts-without-design.md"
    },
    {
      "source": "requirements-ambiguity.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "requirements-ambiguity.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "requirements-ambiguity.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "requirements-ambiguity.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "requirements-ambiguity.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "requirements-ambiguity.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "resistance-to-change.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "resistance-to-change.md",
      "target": "inability-to-innovate.md"
    },
    {
      "source": "resistance-to-change.md",
      "target": "invisible-nature-of-technical-debt.md"
    },
    {
      "source": "resistance-to-change.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "resistance-to-change.md",
      "target": "manual-deployment-processes.md"
    },
    {
      "source": "resistance-to-change.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "resistance-to-change.md",
      "target": "reduced-innovation.md"
    },
    {
      "source": "resistance-to-change.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "resistance-to-change.md",
      "target": "ripple-effect-of-changes.md"
    },
    {
      "source": "resistance-to-change.md",
      "target": "skill-development-gaps.md"
    },
    {
      "source": "resistance-to-change.md",
      "target": "stagnant-architecture.md"
    },
    {
      "source": "resistance-to-change.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "resistance-to-change.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "resource-allocation-failures.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "resource-allocation-failures.md",
      "target": "high-connection-count.md"
    },
    {
      "source": "resource-allocation-failures.md",
      "target": "invisible-nature-of-technical-debt.md"
    },
    {
      "source": "resource-allocation-failures.md",
      "target": "memory-fragmentation.md"
    },
    {
      "source": "resource-allocation-failures.md",
      "target": "memory-leaks.md"
    },
    {
      "source": "resource-allocation-failures.md",
      "target": "resource-contention.md"
    },
    {
      "source": "resource-allocation-failures.md",
      "target": "unreleased-resources.md"
    },
    {
      "source": "resource-contention.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "resource-contention.md",
      "target": "endianness-conversion-overhead.md"
    },
    {
      "source": "resource-contention.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "resource-contention.md",
      "target": "excessive-logging.md"
    },
    {
      "source": "resource-contention.md",
      "target": "false-sharing.md"
    },
    {
      "source": "resource-contention.md",
      "target": "graphql-complexity-issues.md"
    },
    {
      "source": "resource-contention.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "resource-contention.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "resource-contention.md",
      "target": "load-balancing-problems.md"
    },
    {
      "source": "resource-contention.md",
      "target": "lock-contention.md"
    },
    {
      "source": "resource-contention.md",
      "target": "memory-barrier-inefficiency.md"
    },
    {
      "source": "resource-contention.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "resource-contention.md",
      "target": "rate-limiting-issues.md"
    },
    {
      "source": "resource-contention.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "resource-contention.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "resource-contention.md",
      "target": "serialization-deserialization-bottlenecks.md"
    },
    {
      "source": "resource-contention.md",
      "target": "service-timeouts.md"
    },
    {
      "source": "resource-contention.md",
      "target": "shared-database.md"
    },
    {
      "source": "resource-contention.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "resource-contention.md",
      "target": "upstream-timeouts.md"
    },
    {
      "source": "resource-contention.md",
      "target": "virtual-memory-thrashing.md"
    },
    {
      "source": "resource-waste.md",
      "target": "capacity-mismatch.md"
    },
    {
      "source": "resource-waste.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "resource-waste.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "resource-waste.md",
      "target": "modernization-strategy-paralysis.md"
    },
    {
      "source": "resource-waste.md",
      "target": "no-continuous-feedback-loop.md"
    },
    {
      "source": "resource-waste.md",
      "target": "no-formal-change-control-process.md"
    },
    {
      "source": "resource-waste.md",
      "target": "team-confusion.md"
    },
    {
      "source": "resource-waste.md",
      "target": "unused-indexes.md"
    },
    {
      "source": "resource-waste.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "rest-api-design-issues.md",
      "target": "serialization-deserialization-bottlenecks.md"
    },
    {
      "source": "review-bottlenecks.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "review-bottlenecks.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "review-bottlenecks.md",
      "target": "flaky-tests.md"
    },
    {
      "source": "review-bottlenecks.md",
      "target": "large-pull-requests.md"
    },
    {
      "source": "review-bottlenecks.md",
      "target": "long-lived-feature-branches.md"
    },
    {
      "source": "review-bottlenecks.md",
      "target": "perfectionist-review-culture.md"
    },
    {
      "source": "review-bottlenecks.md",
      "target": "rapid-team-growth.md"
    },
    {
      "source": "review-bottlenecks.md",
      "target": "reduced-code-submission-frequency.md"
    },
    {
      "source": "review-bottlenecks.md",
      "target": "reduced-review-participation.md"
    },
    {
      "source": "review-bottlenecks.md",
      "target": "review-process-avoidance.md"
    },
    {
      "source": "review-bottlenecks.md",
      "target": "reviewer-anxiety.md"
    },
    {
      "source": "review-bottlenecks.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "review-bottlenecks.md",
      "target": "slow-development-velocity.md"
    },
    {
      "source": "review-bottlenecks.md",
      "target": "team-members-not-engaged-in-review-process.md"
    },
    {
      "source": "review-bottlenecks.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "review-process-avoidance.md",
      "target": "author-frustration.md"
    },
    {
      "source": "review-process-avoidance.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "review-process-avoidance.md",
      "target": "extended-review-cycles.md"
    },
    {
      "source": "review-process-avoidance.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "review-process-avoidance.md",
      "target": "reduced-code-submission-frequency.md"
    },
    {
      "source": "review-process-avoidance.md",
      "target": "reduced-review-participation.md"
    },
    {
      "source": "review-process-avoidance.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "release-instability.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "review-process-avoidance.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "reviewer-anxiety.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "reviewer-inexperience.md"
    },
    {
      "source": "review-process-breakdown.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "reviewer-anxiety.md",
      "target": "fear-of-conflict.md"
    },
    {
      "source": "reviewer-anxiety.md",
      "target": "reduced-review-participation.md"
    },
    {
      "source": "reviewer-anxiety.md",
      "target": "review-process-avoidance.md"
    },
    {
      "source": "reviewer-anxiety.md",
      "target": "reviewer-inexperience.md"
    },
    {
      "source": "reviewer-inexperience.md",
      "target": "inadequate-code-reviews.md"
    },
    {
      "source": "reviewer-inexperience.md",
      "target": "inadequate-initial-reviews.md"
    },
    {
      "source": "reviewer-inexperience.md",
      "target": "reduced-review-participation.md"
    },
    {
      "source": "reviewer-inexperience.md",
      "target": "review-process-breakdown.md"
    },
    {
      "source": "reviewer-inexperience.md",
      "target": "reviewer-anxiety.md"
    },
    {
      "source": "reviewer-inexperience.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "reviewer-inexperience.md",
      "target": "superficial-code-reviews.md"
    },
    {
      "source": "ripple-effect-of-changes.md",
      "target": "circular-dependency-problems.md"
    },
    {
      "source": "ripple-effect-of-changes.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "ripple-effect-of-changes.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "ripple-effect-of-changes.md",
      "target": "high-coupling-low-cohesion.md"
    },
    {
      "source": "ripple-effect-of-changes.md",
      "target": "poor-domain-model.md"
    },
    {
      "source": "ripple-effect-of-changes.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "ripple-effect-of-changes.md",
      "target": "poorly-defined-responsibilities.md"
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
      "source": "ripple-effect-of-changes.md",
      "target": "tight-coupling-issues.md"
    },
    {
      "source": "rushed-approvals.md",
      "target": "approval-dependencies.md"
    },
    {
      "source": "rushed-approvals.md",
      "target": "fear-of-conflict.md"
    },
    {
      "source": "rushed-approvals.md",
      "target": "large-pull-requests.md"
    },
    {
      "source": "rushed-approvals.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "rushed-approvals.md",
      "target": "reviewer-anxiety.md"
    },
    {
      "source": "rushed-approvals.md",
      "target": "reviewer-inexperience.md"
    },
    {
      "source": "scaling-inefficiencies.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "scaling-inefficiencies.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "scaling-inefficiencies.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "scaling-inefficiencies.md",
      "target": "data-structure-cache-inefficiency.md"
    },
    {
      "source": "scaling-inefficiencies.md",
      "target": "database-query-performance-issues.md"
    },
    {
      "source": "scaling-inefficiencies.md",
      "target": "false-sharing.md"
    },
    {
      "source": "scaling-inefficiencies.md",
      "target": "insufficient-worker-capacity.md"
    },
    {
      "source": "scaling-inefficiencies.md",
      "target": "load-balancing-problems.md"
    },
    {
      "source": "scaling-inefficiencies.md",
      "target": "lock-contention.md"
    },
    {
      "source": "scaling-inefficiencies.md",
      "target": "monolithic-architecture-constraints.md"
    },
    {
      "source": "scaling-inefficiencies.md",
      "target": "resource-contention.md"
    },
    {
      "source": "scaling-inefficiencies.md",
      "target": "shared-database.md"
    },
    {
      "source": "schema-evolution-paralysis.md",
      "target": "shared-database.md"
    },
    {
      "source": "scope-change-resistance.md",
      "target": "poor-contract-design.md"
    },
    {
      "source": "scope-change-resistance.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "scope-creep.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "scope-creep.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "scope-creep.md",
      "target": "deadline-pressure.md"
    },
    {
      "source": "scope-creep.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "scope-creep.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "scope-creep.md",
      "target": "feature-creep.md"
    },
    {
      "source": "scope-creep.md",
      "target": "frequent-changes-to-requirements.md"
    },
    {
      "source": "scope-creep.md",
      "target": "gold-plating.md"
    },
    {
      "source": "scope-creep.md",
      "target": "inadequate-requirements-gathering.md"
    },
    {
      "source": "scope-creep.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "scope-creep.md",
      "target": "no-continuous-feedback-loop.md"
    },
    {
      "source": "scope-creep.md",
      "target": "no-formal-change-control-process.md"
    },
    {
      "source": "scope-creep.md",
      "target": "poor-planning.md"
    },
    {
      "source": "scope-creep.md",
      "target": "rapid-team-growth.md"
    },
    {
      "source": "scope-creep.md",
      "target": "reduced-feature-quality.md"
    },
    {
      "source": "scope-creep.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "scope-creep.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "secret-management-problems.md",
      "target": "environment-variable-issues.md"
    },
    {
      "source": "secret-management-problems.md",
      "target": "password-security-weaknesses.md"
    },
    {
      "source": "serialization-deserialization-bottlenecks.md",
      "target": "endianness-conversion-overhead.md"
    },
    {
      "source": "service-discovery-failures.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "garbage-collection-pressure.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "graphql-complexity-issues.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "high-connection-count.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "incorrect-max-connection-pool-size.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "increased-error-rates.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "insufficient-worker-capacity.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "load-balancing-problems.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "long-running-database-transactions.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "microservice-communication-overhead.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "misconfigured-connection-pools.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "network-latency.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "unreleased-resources.md"
    },
    {
      "source": "service-timeouts.md",
      "target": "virtual-memory-thrashing.md"
    },
    {
      "source": "session-management-issues.md",
      "target": "cross-site-scripting-vulnerabilities.md"
    },
    {
      "source": "session-management-issues.md",
      "target": "password-security-weaknesses.md"
    },
    {
      "source": "shadow-systems.md",
      "target": "feature-gaps.md"
    },
    {
      "source": "shadow-systems.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "shared-database.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "shared-database.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "shared-database.md",
      "target": "schema-evolution-paralysis.md"
    },
    {
      "source": "shared-dependencies.md",
      "target": "dependency-version-conflicts.md"
    },
    {
      "source": "shared-dependencies.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "cv-driven-development.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "delayed-bug-fixes.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "dependency-on-supplier.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "excessive-class-size.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "feature-factory.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "inadequate-mentoring-structure.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "inadequate-test-data-management.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "inadequate-test-infrastructure.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "insufficient-testing.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "invisible-nature-of-technical-debt.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "legacy-code-without-tests.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "monolithic-architecture-constraints.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "monolithic-functions-and-classes.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "obsolete-technologies.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "shared-database.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "stagnant-architecture.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "test-debt.md"
    },
    {
      "source": "short-term-focus.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "silent-data-corruption.md",
      "target": "data-migration-integrity-issues.md"
    },
    {
      "source": "silent-data-corruption.md",
      "target": "dma-coherency-issues.md"
    },
    {
      "source": "silent-data-corruption.md",
      "target": "insecure-data-transmission.md"
    },
    {
      "source": "silent-data-corruption.md",
      "target": "integer-overflow-underflow.md"
    },
    {
      "source": "silent-data-corruption.md",
      "target": "null-pointer-dereferences.md"
    },
    {
      "source": "silent-data-corruption.md",
      "target": "race-conditions.md"
    },
    {
      "source": "single-points-of-failure.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "single-points-of-failure.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "single-points-of-failure.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "single-points-of-failure.md",
      "target": "knowledge-silos.md"
    },
    {
      "source": "single-points-of-failure.md",
      "target": "legacy-skill-shortage.md"
    },
    {
      "source": "single-points-of-failure.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "single-points-of-failure.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "single-points-of-failure.md",
      "target": "skill-development-gaps.md"
    },
    {
      "source": "single-points-of-failure.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "skill-development-gaps.md",
      "target": "inadequate-mentoring-structure.md"
    },
    {
      "source": "skill-development-gaps.md",
      "target": "inexperienced-developers.md"
    },
    {
      "source": "skill-development-gaps.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "skill-development-gaps.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "skill-development-gaps.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "skill-development-gaps.md",
      "target": "procedural-background.md"
    },
    {
      "source": "skill-development-gaps.md",
      "target": "queries-that-prevent-index-usage.md"
    },
    {
      "source": "skill-development-gaps.md",
      "target": "race-conditions.md"
    },
    {
      "source": "skill-development-gaps.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "skill-development-gaps.md",
      "target": "uneven-workload-distribution.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "alignment-and-padding-issues.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "atomic-operation-overhead.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "data-structure-cache-inefficiency.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "database-query-performance-issues.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "endianness-conversion-overhead.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "excessive-disk-io.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "excessive-logging.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "excessive-object-allocation.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "false-sharing.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "feature-bloat.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "garbage-collection-pressure.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "graphql-complexity-issues.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "high-client-side-resource-consumption.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "high-connection-count.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "high-number-of-database-queries.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "high-resource-utilization-on-client.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "imperative-data-fetching-logic.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "improper-event-listener-management.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "index-fragmentation.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "inefficient-code.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "inefficient-database-indexing.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "inefficient-frontend-code.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "insufficient-worker-capacity.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "interrupt-overhead.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "load-balancing-problems.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "lock-contention.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "log-spam.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "memory-barrier-inefficiency.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "memory-fragmentation.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "memory-swapping.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "microservice-communication-overhead.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "n-plus-one-query-problem.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "negative-brand-perception.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "negative-user-feedback.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "network-latency.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "poor-caching-strategy.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "serialization-deserialization-bottlenecks.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "service-timeouts.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "slow-database-queries.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "unbounded-data-growth.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "unbounded-data-structures.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "unoptimized-file-access.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "user-frustration.md"
    },
    {
      "source": "slow-application-performance.md",
      "target": "virtual-memory-thrashing.md"
    },
    {
      "source": "slow-database-queries.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "slow-database-queries.md",
      "target": "index-fragmentation.md"
    },
    {
      "source": "slow-database-queries.md",
      "target": "inefficient-database-indexing.md"
    },
    {
      "source": "slow-database-queries.md",
      "target": "lazy-loading.md"
    },
    {
      "source": "slow-database-queries.md",
      "target": "long-running-database-transactions.md"
    },
    {
      "source": "slow-database-queries.md",
      "target": "long-running-transactions.md"
    },
    {
      "source": "slow-database-queries.md",
      "target": "queries-that-prevent-index-usage.md"
    },
    {
      "source": "slow-database-queries.md",
      "target": "slow-application-performance.md"
    },
    {
      "source": "slow-database-queries.md",
      "target": "slow-response-times-for-lists.md"
    },
    {
      "source": "slow-database-queries.md",
      "target": "unused-indexes.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "analysis-paralysis.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "context-switching-overhead.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "defensive-coding-practices.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "difficult-code-comprehension.md"
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
      "target": "duplicated-effort.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "duplicated-research-effort.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "extended-research-time.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "flaky-tests.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "high-bug-introduction-rate.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "high-turnover.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "implementation-rework.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "inadequate-initial-reviews.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "inappropriate-skillset.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "increased-cognitive-load.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "increased-manual-testing-effort.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "increased-manual-work.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "knowledge-dependency.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "knowledge-gaps.md"
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
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "mental-fatigue.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "merge-conflicts.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "micromanagement-culture.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "reduced-individual-productivity.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "reduced-team-productivity.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "release-anxiety.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "scope-creep.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "staff-availability-issues.md"
    },
    {
      "source": "slow-development-velocity.md",
      "target": "stakeholder-dissatisfaction.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "bloated-class.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "complex-implementation-paths.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "constant-firefighting.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "database-schema-design-problems.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "feature-creep.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "god-object-anti-pattern.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "hardcoded-values.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "high-coupling-low-cohesion.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "increased-time-to-market.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "inefficient-development-environment.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "information-decay.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "integration-difficulties.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "legacy-api-versioning-nightmare.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "monolithic-architecture-constraints.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "new-hire-frustration.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "nitpicking-culture.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "rest-api-design-issues.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "ripple-effect-of-changes.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "schema-evolution-paralysis.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "single-entry-point-design.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "slow-feature-development.md",
      "target": "stagnant-architecture.md"
    },
    {
      "source": "slow-incident-resolution.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "slow-incident-resolution.md",
      "target": "inadequate-configuration-management.md"
    },
    {
      "source": "slow-incident-resolution.md",
      "target": "inadequate-error-handling.md"
    },
    {
      "source": "slow-incident-resolution.md",
      "target": "insufficient-audit-logging.md"
    },
    {
      "source": "slow-incident-resolution.md",
      "target": "knowledge-silos.md"
    },
    {
      "source": "slow-incident-resolution.md",
      "target": "legacy-configuration-management-chaos.md"
    },
    {
      "source": "slow-incident-resolution.md",
      "target": "legacy-skill-shortage.md"
    },
    {
      "source": "slow-incident-resolution.md",
      "target": "log-spam.md"
    },
    {
      "source": "slow-incident-resolution.md",
      "target": "missing-end-to-end-tests.md"
    },
    {
      "source": "slow-incident-resolution.md",
      "target": "missing-rollback-strategy.md"
    },
    {
      "source": "slow-incident-resolution.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "slow-incident-resolution.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "slow-incident-resolution.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "slow-incident-resolution.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "slow-knowledge-transfer.md",
      "target": "clever-code.md"
    },
    {
      "source": "slow-knowledge-transfer.md",
      "target": "language-barriers.md"
    },
    {
      "source": "slow-knowledge-transfer.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "slow-knowledge-transfer.md",
      "target": "rapid-team-growth.md"
    },
    {
      "source": "slow-knowledge-transfer.md",
      "target": "skill-development-gaps.md"
    },
    {
      "source": "slow-knowledge-transfer.md",
      "target": "tacit-knowledge.md"
    },
    {
      "source": "slow-response-times-for-lists.md",
      "target": "incorrect-index-type.md"
    },
    {
      "source": "slow-response-times-for-lists.md",
      "target": "poor-caching-strategy.md"
    },
    {
      "source": "slow-response-times-for-lists.md",
      "target": "resource-contention.md"
    },
    {
      "source": "slow-response-times-for-lists.md",
      "target": "slow-database-queries.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "debugging-difficulties.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "difficult-code-comprehension.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "feature-creep-without-refactoring.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "global-state-and-side-effects.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "implementation-starts-without-design.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "insufficient-design-skills.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "legacy-business-logic-extraction-difficulty.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "maintenance-overhead.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "misunderstanding-of-oop.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "procedural-programming-in-oop-languages.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "slow-development-velocity.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "slow-feature-development.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "stack-overflow-errors.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "spaghetti-code.md",
      "target": "unpredictable-system-behavior.md"
    },
    {
      "source": "sql-injection-vulnerabilities.md",
      "target": "error-message-information-disclosure.md"
    },
    {
      "source": "staff-availability-issues.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "staff-availability-issues.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "staff-availability-issues.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "staff-availability-issues.md",
      "target": "single-points-of-failure.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "architectural-mismatch.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "decision-avoidance.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "fear-of-change.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "history-of-failed-changes.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "implementation-starts-without-design.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "inability-to-innovate.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "integration-difficulties.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "reduced-innovation.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "regulatory-compliance-drift.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "schema-evolution-paralysis.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "short-term-focus.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "technical-architecture-limitations.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "technology-isolation.md"
    },
    {
      "source": "stagnant-architecture.md",
      "target": "technology-stack-fragmentation.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "communication-risk-outside-project.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "data-protection-risk.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "declining-business-metrics.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "planning-credibility-issues.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "poor-project-control.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "reduced-predictability.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "release-instability.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "stakeholder-developer-communication-gap.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "stakeholder-dissatisfaction.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "stakeholder-frustration.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "strangler-fig-pattern-failures.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "system-outages.md"
    },
    {
      "source": "stakeholder-confidence-loss.md",
      "target": "user-trust-erosion.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "assumption-based-development.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "complex-domain-model.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "feature-gaps.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "feedback-isolation.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "frequent-changes-to-requirements.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "implementation-rework.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "inadequate-requirements-gathering.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "invisible-nature-of-technical-debt.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "modernization-roi-justification-failure.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "no-continuous-feedback-loop.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "poor-contract-design.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "poor-domain-model.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "poor-planning.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "stakeholder-confidence-loss.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "stakeholder-dissatisfaction.md"
    },
    {
      "source": "stakeholder-developer-communication-gap.md",
      "target": "stakeholder-frustration.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "breaking-changes.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "budget-overruns.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "cascade-delays.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "communication-risk-outside-project.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "feedback-isolation.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "legal-disputes.md"
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
      "target": "missing-end-to-end-tests.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "project-authority-vacuum.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "quality-blind-spots.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "quality-degradation.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "reduced-team-productivity.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "scope-change-resistance.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "stakeholder-developer-communication-gap.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "stakeholder-dissatisfaction.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "analysis-paralysis.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "communication-risk-outside-project.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "delayed-value-delivery.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "extended-cycle-times.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "large-estimates-for-small-changes.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "modernization-roi-justification-failure.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "negative-user-feedback.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "no-continuous-feedback-loop.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "planning-credibility-issues.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "resource-waste.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "schema-evolution-paralysis.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "stakeholder-frustration.md",
      "target": "user-frustration.md"
    },
    {
      "source": "style-arguments-in-code-reviews.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "style-arguments-in-code-reviews.md",
      "target": "inconsistent-codebase.md"
    },
    {
      "source": "style-arguments-in-code-reviews.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "style-arguments-in-code-reviews.md",
      "target": "undefined-code-style-guidelines.md"
    },
    {
      "source": "suboptimal-solutions.md",
      "target": "accumulated-decision-debt.md"
    },
    {
      "source": "suboptimal-solutions.md",
      "target": "knowledge-gaps.md"
    },
    {
      "source": "suboptimal-solutions.md",
      "target": "poor-communication.md"
    },
    {
      "source": "suboptimal-solutions.md",
      "target": "scope-change-resistance.md"
    },
    {
      "source": "superficial-code-reviews.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "superficial-code-reviews.md",
      "target": "fear-of-conflict.md"
    },
    {
      "source": "superficial-code-reviews.md",
      "target": "large-pull-requests.md"
    },
    {
      "source": "superficial-code-reviews.md",
      "target": "lower-code-quality.md"
    },
    {
      "source": "superficial-code-reviews.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "superficial-code-reviews.md",
      "target": "review-process-breakdown.md"
    },
    {
      "source": "superficial-code-reviews.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "superficial-code-reviews.md",
      "target": "time-pressure.md"
    },
    {
      "source": "synchronization-problems.md",
      "target": "code-duplication.md"
    },
    {
      "source": "synchronization-problems.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "synchronization-problems.md",
      "target": "incomplete-knowledge.md"
    },
    {
      "source": "synchronization-problems.md",
      "target": "race-conditions.md"
    },
    {
      "source": "system-integration-blindness.md",
      "target": "hidden-dependencies.md"
    },
    {
      "source": "system-outages.md",
      "target": "authentication-bypass-vulnerabilities.md"
    },
    {
      "source": "system-outages.md",
      "target": "configuration-chaos.md"
    },
    {
      "source": "system-outages.md",
      "target": "cross-system-data-synchronization-problems.md"
    },
    {
      "source": "system-outages.md",
      "target": "customer-dissatisfaction.md"
    },
    {
      "source": "system-outages.md",
      "target": "data-migration-complexities.md"
    },
    {
      "source": "system-outages.md",
      "target": "data-migration-integrity-issues.md"
    },
    {
      "source": "system-outages.md",
      "target": "database-connection-leaks.md"
    },
    {
      "source": "system-outages.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "system-outages.md",
      "target": "deployment-risk.md"
    },
    {
      "source": "system-outages.md",
      "target": "high-database-resource-utilization.md"
    },
    {
      "source": "system-outages.md",
      "target": "immature-delivery-strategy.md"
    },
    {
      "source": "system-outages.md",
      "target": "inadequate-configuration-management.md"
    },
    {
      "source": "system-outages.md",
      "target": "inadequate-error-handling.md"
    },
    {
      "source": "system-outages.md",
      "target": "large-risky-releases.md"
    },
    {
      "source": "system-outages.md",
      "target": "load-balancing-problems.md"
    },
    {
      "source": "system-outages.md",
      "target": "log-injection-vulnerabilities.md"
    },
    {
      "source": "system-outages.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "system-outages.md",
      "target": "negative-brand-perception.md"
    },
    {
      "source": "system-outages.md",
      "target": "negative-user-feedback.md"
    },
    {
      "source": "system-outages.md",
      "target": "null-pointer-dereferences.md"
    },
    {
      "source": "system-outages.md",
      "target": "operational-overhead.md"
    },
    {
      "source": "system-outages.md",
      "target": "poor-operational-concept.md"
    },
    {
      "source": "system-outages.md",
      "target": "poor-system-environment.md"
    },
    {
      "source": "system-outages.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "system-outages.md",
      "target": "service-discovery-failures.md"
    },
    {
      "source": "system-outages.md",
      "target": "sql-injection-vulnerabilities.md"
    },
    {
      "source": "system-outages.md",
      "target": "stack-overflow-errors.md"
    },
    {
      "source": "system-outages.md",
      "target": "strangler-fig-pattern-failures.md"
    },
    {
      "source": "system-outages.md",
      "target": "system-integration-blindness.md"
    },
    {
      "source": "system-outages.md",
      "target": "task-queues-backing-up.md"
    },
    {
      "source": "system-outages.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "system-outages.md",
      "target": "virtual-memory-thrashing.md"
    },
    {
      "source": "system-stagnation.md",
      "target": "difficulty-quantifying-benefits.md"
    },
    {
      "source": "system-stagnation.md",
      "target": "fear-of-breaking-changes.md"
    },
    {
      "source": "system-stagnation.md",
      "target": "maintenance-paralysis.md"
    },
    {
      "source": "system-stagnation.md",
      "target": "obsolete-technologies.md"
    },
    {
      "source": "system-stagnation.md",
      "target": "past-negative-experiences.md"
    },
    {
      "source": "system-stagnation.md",
      "target": "resistance-to-change.md"
    },
    {
      "source": "system-stagnation.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "tacit-knowledge.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "tacit-knowledge.md",
      "target": "knowledge-silos.md"
    },
    {
      "source": "tacit-knowledge.md",
      "target": "slow-knowledge-transfer.md"
    },
    {
      "source": "tangled-cross-cutting-concerns.md",
      "target": "excessive-logging.md"
    },
    {
      "source": "task-queues-backing-up.md",
      "target": "inefficient-code.md"
    },
    {
      "source": "team-churn-impact.md",
      "target": "inconsistent-codebase.md"
    },
    {
      "source": "team-churn-impact.md",
      "target": "lack-of-ownership-and-accountability.md"
    },
    {
      "source": "team-churn-impact.md",
      "target": "lower-code-quality.md"
    },
    {
      "source": "team-churn-impact.md",
      "target": "undefined-code-style-guidelines.md"
    },
    {
      "source": "team-confusion.md",
      "target": "changing-project-scope.md"
    },
    {
      "source": "team-confusion.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "team-confusion.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "team-confusion.md",
      "target": "duplicated-work.md"
    },
    {
      "source": "team-confusion.md",
      "target": "frequent-changes-to-requirements.md"
    },
    {
      "source": "team-confusion.md",
      "target": "inconsistent-execution.md"
    },
    {
      "source": "team-confusion.md",
      "target": "language-barriers.md"
    },
    {
      "source": "team-confusion.md",
      "target": "power-struggles.md"
    },
    {
      "source": "team-confusion.md",
      "target": "product-direction-chaos.md"
    },
    {
      "source": "team-confusion.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "team-confusion.md",
      "target": "unclear-goals-and-priorities.md"
    },
    {
      "source": "team-confusion.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "change-management-chaos.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "duplicated-work.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "merge-conflicts.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "organizational-structure-mismatch.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "shared-database.md"
    },
    {
      "source": "team-coordination-issues.md",
      "target": "team-silos.md"
    },
    {
      "source": "team-demoralization.md",
      "target": "changing-project-scope.md"
    },
    {
      "source": "team-demoralization.md",
      "target": "competing-priorities.md"
    },
    {
      "source": "team-demoralization.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "team-demoralization.md",
      "target": "decision-avoidance.md"
    },
    {
      "source": "team-demoralization.md",
      "target": "decision-paralysis.md"
    },
    {
      "source": "team-demoralization.md",
      "target": "developer-frustration-and-burnout.md"
    },
    {
      "source": "team-demoralization.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "team-demoralization.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "team-demoralization.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "team-demoralization.md",
      "target": "planning-credibility-issues.md"
    },
    {
      "source": "team-demoralization.md",
      "target": "team-confusion.md"
    },
    {
      "source": "team-demoralization.md",
      "target": "team-dysfunction.md"
    },
    {
      "source": "team-demoralization.md",
      "target": "unclear-goals-and-priorities.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "author-frustration.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "individual-recognition-culture.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "poor-communication.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "power-struggles.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "reduced-team-productivity.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "team-demoralization.md"
    },
    {
      "source": "team-dysfunction.md",
      "target": "team-members-not-engaged-in-review-process.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "insufficient-code-review.md"
    },
    {
      "source": "team-members-not-engaged-in-review-process.md",
      "target": "reduced-review-participation.md"
    },
    {
      "source": "team-silos.md",
      "target": "code-duplication.md"
    },
    {
      "source": "team-silos.md",
      "target": "communication-breakdown.md"
    },
    {
      "source": "team-silos.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "team-silos.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "team-silos.md",
      "target": "duplicated-research-effort.md"
    },
    {
      "source": "team-silos.md",
      "target": "feedback-isolation.md"
    },
    {
      "source": "team-silos.md",
      "target": "inadequate-integration-tests.md"
    },
    {
      "source": "team-silos.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "team-silos.md",
      "target": "knowledge-silos.md"
    },
    {
      "source": "team-silos.md",
      "target": "language-barriers.md"
    },
    {
      "source": "team-silos.md",
      "target": "no-continuous-feedback-loop.md"
    },
    {
      "source": "team-silos.md",
      "target": "poor-communication.md"
    },
    {
      "source": "team-silos.md",
      "target": "poor-interfaces-between-applications.md"
    },
    {
      "source": "team-silos.md",
      "target": "system-integration-blindness.md"
    },
    {
      "source": "team-silos.md",
      "target": "team-coordination-issues.md"
    },
    {
      "source": "team-silos.md",
      "target": "team-dysfunction.md"
    },
    {
      "source": "team-silos.md",
      "target": "team-members-not-engaged-in-review-process.md"
    },
    {
      "source": "team-silos.md",
      "target": "technology-stack-fragmentation.md"
    },
    {
      "source": "team-silos.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "competitive-disadvantage.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "endianness-conversion-overhead.md"
    },
    {
      "source": "technical-architecture-limitations.md",
      "target": "tool-limitations.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "integration-difficulties.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "legacy-skill-shortage.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "obsolete-technologies.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "vendor-dependency-entrapment.md"
    },
    {
      "source": "technology-isolation.md",
      "target": "vendor-lock-in.md"
    },
    {
      "source": "technology-lock-in.md",
      "target": "dependency-on-supplier.md"
    },
    {
      "source": "technology-lock-in.md",
      "target": "monolithic-architecture-constraints.md"
    },
    {
      "source": "technology-lock-in.md",
      "target": "obsolete-technologies.md"
    },
    {
      "source": "technology-lock-in.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "technology-lock-in.md",
      "target": "shared-dependencies.md"
    },
    {
      "source": "technology-lock-in.md",
      "target": "system-stagnation.md"
    },
    {
      "source": "technology-lock-in.md",
      "target": "technical-architecture-limitations.md"
    },
    {
      "source": "technology-lock-in.md",
      "target": "technology-isolation.md"
    },
    {
      "source": "technology-stack-fragmentation.md",
      "target": "context-switching-overhead.md"
    },
    {
      "source": "technology-stack-fragmentation.md",
      "target": "cv-driven-development.md"
    },
    {
      "source": "technology-stack-fragmentation.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "technology-stack-fragmentation.md",
      "target": "shadow-systems.md"
    },
    {
      "source": "test-debt.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "test-debt.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "test-debt.md",
      "target": "regression-bugs.md"
    },
    {
      "source": "test-debt.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "test-debt.md",
      "target": "testing-environment-fragility.md"
    },
    {
      "source": "test-debt.md",
      "target": "time-pressure.md"
    },
    {
      "source": "testing-complexity.md",
      "target": "code-duplication.md"
    },
    {
      "source": "testing-complexity.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "testing-complexity.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "testing-complexity.md",
      "target": "missing-end-to-end-tests.md"
    },
    {
      "source": "testing-environment-fragility.md",
      "target": "flaky-tests.md"
    },
    {
      "source": "thread-pool-exhaustion.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "thread-pool-exhaustion.md",
      "target": "lock-contention.md"
    },
    {
      "source": "thread-pool-exhaustion.md",
      "target": "resource-allocation-failures.md"
    },
    {
      "source": "thread-pool-exhaustion.md",
      "target": "service-timeouts.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "brittle-codebase.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "cache-invalidation-problems.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "cascade-failures.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "circular-dependency-problems.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "cognitive-overload.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "complex-implementation-paths.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "deployment-coupling.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "difficult-code-reuse.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "difficult-to-test-code.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "flaky-tests.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "high-maintenance-costs.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "implementation-starts-without-design.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "increased-risk-of-bugs.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "integration-difficulties.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "monolithic-architecture-constraints.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "poor-encapsulation.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "regression-bugs.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "ripple-effect-of-changes.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "scaling-inefficiencies.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "shared-database.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "team-coordination-issues.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "technology-lock-in.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "testing-complexity.md"
    },
    {
      "source": "tight-coupling-issues.md",
      "target": "unpredictable-system-behavior.md"
    },
    {
      "source": "time-pressure.md",
      "target": "bloated-class.md"
    },
    {
      "source": "time-pressure.md",
      "target": "cargo-culting.md"
    },
    {
      "source": "time-pressure.md",
      "target": "code-duplication.md"
    },
    {
      "source": "time-pressure.md",
      "target": "convenience-driven-development.md"
    },
    {
      "source": "time-pressure.md",
      "target": "copy-paste-programming.md"
    },
    {
      "source": "time-pressure.md",
      "target": "high-technical-debt.md"
    },
    {
      "source": "time-pressure.md",
      "target": "implementation-starts-without-design.md"
    },
    {
      "source": "time-pressure.md",
      "target": "implicit-knowledge.md"
    },
    {
      "source": "time-pressure.md",
      "target": "inadequate-code-reviews.md"
    },
    {
      "source": "time-pressure.md",
      "target": "inadequate-error-handling.md"
    },
    {
      "source": "time-pressure.md",
      "target": "inadequate-initial-reviews.md"
    },
    {
      "source": "time-pressure.md",
      "target": "inadequate-integration-tests.md"
    },
    {
      "source": "time-pressure.md",
      "target": "inadequate-requirements-gathering.md"
    },
    {
      "source": "time-pressure.md",
      "target": "increased-stress-and-burnout.md"
    },
    {
      "source": "time-pressure.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "time-pressure.md",
      "target": "information-decay.md"
    },
    {
      "source": "time-pressure.md",
      "target": "limited-team-learning.md"
    },
    {
      "source": "time-pressure.md",
      "target": "lower-code-quality.md"
    },
    {
      "source": "time-pressure.md",
      "target": "monitoring-gaps.md"
    },
    {
      "source": "time-pressure.md",
      "target": "no-continuous-feedback-loop.md"
    },
    {
      "source": "time-pressure.md",
      "target": "outdated-tests.md"
    },
    {
      "source": "time-pressure.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "time-pressure.md",
      "target": "procrastination-on-complex-tasks.md"
    },
    {
      "source": "time-pressure.md",
      "target": "project-resource-constraints.md"
    },
    {
      "source": "time-pressure.md",
      "target": "quality-compromises.md"
    },
    {
      "source": "time-pressure.md",
      "target": "review-bottlenecks.md"
    },
    {
      "source": "time-pressure.md",
      "target": "review-process-avoidance.md"
    },
    {
      "source": "time-pressure.md",
      "target": "review-process-breakdown.md"
    },
    {
      "source": "time-pressure.md",
      "target": "rushed-approvals.md"
    },
    {
      "source": "time-pressure.md",
      "target": "skill-development-gaps.md"
    },
    {
      "source": "time-pressure.md",
      "target": "spaghetti-code.md"
    },
    {
      "source": "time-pressure.md",
      "target": "stagnant-architecture.md"
    },
    {
      "source": "time-pressure.md",
      "target": "test-debt.md"
    },
    {
      "source": "time-pressure.md",
      "target": "unclear-documentation-ownership.md"
    },
    {
      "source": "time-pressure.md",
      "target": "uncontrolled-codebase-growth.md"
    },
    {
      "source": "time-pressure.md",
      "target": "undefined-code-style-guidelines.md"
    },
    {
      "source": "time-pressure.md",
      "target": "unrealistic-deadlines.md"
    },
    {
      "source": "time-pressure.md",
      "target": "unrealistic-schedule.md"
    },
    {
      "source": "time-pressure.md",
      "target": "workaround-culture.md"
    },
    {
      "source": "tool-limitations.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "tool-limitations.md",
      "target": "inefficient-development-environment.md"
    },
    {
      "source": "tool-limitations.md",
      "target": "information-fragmentation.md"
    },
    {
      "source": "tool-limitations.md",
      "target": "reduced-individual-productivity.md"
    },
    {
      "source": "tool-limitations.md",
      "target": "unoptimized-file-access.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "index-fragmentation.md"
    },
    {
      "source": "unbounded-data-growth.md",
      "target": "virtual-memory-thrashing.md"
    },
    {
      "source": "unbounded-data-structures.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "information-decay.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "legacy-system-documentation-archaeology.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "poor-documentation.md"
    },
    {
      "source": "unclear-documentation-ownership.md",
      "target": "team-confusion.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "analysis-paralysis.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "competing-priorities.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "eager-to-please-stakeholders.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "feature-factory.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "poor-teamwork.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "power-struggles.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "priority-thrashing.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "product-direction-chaos.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "project-authority-vacuum.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "resource-waste.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "team-confusion.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "team-demoralization.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "team-dysfunction.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "unclear-sharing-expectations.md"
    },
    {
      "source": "unclear-goals-and-priorities.md",
      "target": "unmotivated-employees.md"
    },
    {
      "source": "unclear-sharing-expectations.md",
      "target": "communication-risk-within-project.md"
    },
    {
      "source": "unclear-sharing-expectations.md",
      "target": "duplicated-research-effort.md"
    },
    {
      "source": "unclear-sharing-expectations.md",
      "target": "knowledge-sharing-breakdown.md"
    },
    {
      "source": "uncontrolled-codebase-growth.md",
      "target": "long-build-and-test-times.md"
    },
    {
      "source": "uncontrolled-codebase-growth.md",
      "target": "monolithic-architecture-constraints.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "author-frustration.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "automated-tooling-ineffectiveness.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "code-review-inefficiency.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "conflicting-reviewer-opinions.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "defensive-coding-practices.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "extended-review-cycles.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "inconsistent-codebase.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "inconsistent-coding-standards.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "inconsistent-naming-conventions.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "mixed-coding-styles.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "nitpicking-culture.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "perfectionist-review-culture.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "poor-naming-conventions.md"
    },
    {
      "source": "undefined-code-style-guidelines.md",
      "target": "style-arguments-in-code-reviews.md"
    },
    {
      "source": "uneven-work-flow.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "uneven-work-flow.md",
      "target": "work-queue-buildup.md"
    },
    {
      "source": "uneven-workload-distribution.md",
      "target": "inconsistent-knowledge-acquisition.md"
    },
    {
      "source": "uneven-workload-distribution.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "uneven-workload-distribution.md",
      "target": "unmotivated-employees.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "team-demoralization.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "unclear-goals-and-priorities.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "unproductive-meetings.md"
    },
    {
      "source": "unmotivated-employees.md",
      "target": "wasted-development-effort.md"
    },
    {
      "source": "unoptimized-file-access.md",
      "target": "inefficient-code.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "configuration-drift.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "global-state-and-side-effects.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "hidden-dependencies.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "hidden-side-effects.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "increasing-brittleness.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "integer-overflow-underflow.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "null-pointer-dereferences.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "resource-contention.md"
    },
    {
      "source": "unpredictable-system-behavior.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "unproductive-meetings.md",
      "target": "bikeshedding.md"
    },
    {
      "source": "unproductive-meetings.md",
      "target": "mental-fatigue.md"
    },
    {
      "source": "unrealistic-deadlines.md",
      "target": "constantly-shifting-deadlines.md"
    },
    {
      "source": "unrealistic-deadlines.md",
      "target": "deadline-pressure.md"
    },
    {
      "source": "unrealistic-deadlines.md",
      "target": "developer-frustration-and-burnout.md"
    },
    {
      "source": "unrealistic-deadlines.md",
      "target": "missed-deadlines.md"
    },
    {
      "source": "unrealistic-deadlines.md",
      "target": "overworked-teams.md"
    },
    {
      "source": "unrealistic-deadlines.md",
      "target": "planning-credibility-issues.md"
    },
    {
      "source": "unrealistic-deadlines.md",
      "target": "time-pressure.md"
    },
    {
      "source": "unrealistic-deadlines.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "unrealistic-schedule.md",
      "target": "delayed-project-timelines.md"
    },
    {
      "source": "unrealistic-schedule.md",
      "target": "rapid-prototyping-becoming-production.md"
    },
    {
      "source": "unrealistic-schedule.md",
      "target": "rapid-team-growth.md"
    },
    {
      "source": "unrealistic-schedule.md",
      "target": "reduced-feature-quality.md"
    },
    {
      "source": "unrealistic-schedule.md",
      "target": "refactoring-avoidance.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "memory-leaks.md"
    },
    {
      "source": "unreleased-resources.md",
      "target": "thread-pool-exhaustion.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "external-service-delays.md"
    },
    {
      "source": "upstream-timeouts.md",
      "target": "rate-limiting-issues.md"
    },
    {
      "source": "user-confusion.md",
      "target": "feature-creep.md"
    },
    {
      "source": "user-confusion.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "user-confusion.md",
      "target": "inconsistent-quality.md"
    },
    {
      "source": "user-confusion.md",
      "target": "user-frustration.md"
    },
    {
      "source": "user-confusion.md",
      "target": "user-trust-erosion.md"
    },
    {
      "source": "user-frustration.md",
      "target": "algorithmic-complexity-problems.md"
    },
    {
      "source": "user-frustration.md",
      "target": "deadlock-conditions.md"
    },
    {
      "source": "user-frustration.md",
      "target": "delayed-bug-fixes.md"
    },
    {
      "source": "user-frustration.md",
      "target": "delayed-issue-resolution.md"
    },
    {
      "source": "user-frustration.md",
      "target": "gradual-performance-degradation.md"
    },
    {
      "source": "user-frustration.md",
      "target": "growing-task-queues.md"
    },
    {
      "source": "user-frustration.md",
      "target": "high-api-latency.md"
    },
    {
      "source": "user-frustration.md",
      "target": "high-client-side-resource-consumption.md"
    },
    {
      "source": "user-frustration.md",
      "target": "inadequate-onboarding.md"
    },
    {
      "source": "user-frustration.md",
      "target": "partial-bug-fixes.md"
    },
    {
      "source": "user-frustration.md",
      "target": "poor-user-experience-ux-design.md"
    },
    {
      "source": "user-frustration.md",
      "target": "suboptimal-solutions.md"
    },
    {
      "source": "user-frustration.md",
      "target": "upstream-timeouts.md"
    },
    {
      "source": "user-frustration.md",
      "target": "user-confusion.md"
    },
    {
      "source": "user-frustration.md",
      "target": "user-trust-erosion.md"
    },
    {
      "source": "user-trust-erosion.md",
      "target": "authentication-bypass-vulnerabilities.md"
    },
    {
      "source": "user-trust-erosion.md",
      "target": "authorization-flaws.md"
    },
    {
      "source": "user-trust-erosion.md",
      "target": "cross-site-scripting-vulnerabilities.md"
    },
    {
      "source": "user-trust-erosion.md",
      "target": "frequent-hotfixes-and-rollbacks.md"
    },
    {
      "source": "user-trust-erosion.md",
      "target": "inconsistent-behavior.md"
    },
    {
      "source": "user-trust-erosion.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "user-trust-erosion.md",
      "target": "release-instability.md"
    },
    {
      "source": "user-trust-erosion.md",
      "target": "session-management-issues.md"
    },
    {
      "source": "user-trust-erosion.md",
      "target": "silent-data-corruption.md"
    },
    {
      "source": "user-trust-erosion.md",
      "target": "slow-incident-resolution.md"
    },
    {
      "source": "user-trust-erosion.md",
      "target": "user-confusion.md"
    },
    {
      "source": "user-trust-erosion.md",
      "target": "user-frustration.md"
    },
    {
      "source": "vendor-dependency-entrapment.md",
      "target": "regulatory-compliance-drift.md"
    },
    {
      "source": "vendor-dependency-entrapment.md",
      "target": "vendor-dependency.md"
    },
    {
      "source": "vendor-dependency-entrapment.md",
      "target": "vendor-lock-in.md"
    },
    {
      "source": "vendor-dependency.md",
      "target": "vendor-dependency-entrapment.md"
    },
    {
      "source": "vendor-dependency.md",
      "target": "vendor-relationship-strain.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "vendor-dependency-entrapment.md"
    },
    {
      "source": "vendor-lock-in.md",
      "target": "vendor-dependency.md"
    },
    {
      "source": "vendor-relationship-strain.md",
      "target": "legal-disputes.md"
    },
    {
      "source": "vendor-relationship-strain.md",
      "target": "poor-contract-design.md"
    },
    {
      "source": "vendor-relationship-strain.md",
      "target": "vendor-dependency.md"
    },
    {
      "source": "virtual-memory-thrashing.md",
      "target": "memory-fragmentation.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "analysis-paralysis.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "assumption-based-development.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "changing-project-scope.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "duplicated-effort.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "duplicated-research-effort.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "duplicated-work.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "feature-factory.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "feedback-isolation.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "frequent-changes-to-requirements.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "implementation-rework.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "incomplete-projects.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "inefficient-processes.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "information-fragmentation.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "misaligned-deliverables.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "planning-dysfunction.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "power-struggles.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "premature-technology-introduction.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "priority-thrashing.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "process-design-flaws.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "product-direction-chaos.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "requirements-ambiguity.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "scope-change-resistance.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "second-system-effect.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "unclear-goals-and-priorities.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "unproductive-meetings.md"
    },
    {
      "source": "wasted-development-effort.md",
      "target": "work-blocking.md"
    },
    {
      "source": "work-blocking.md",
      "target": "decision-avoidance.md"
    },
    {
      "source": "work-blocking.md",
      "target": "delayed-decision-making.md"
    },
    {
      "source": "work-blocking.md",
      "target": "maintenance-bottlenecks.md"
    },
    {
      "source": "work-queue-buildup.md",
      "target": "avoidance-behaviors.md"
    },
    {
      "source": "work-queue-buildup.md",
      "target": "bottleneck-formation.md"
    },
    {
      "source": "work-queue-buildup.md",
      "target": "capacity-mismatch.md"
    },
    {
      "source": "work-queue-buildup.md",
      "target": "uneven-work-flow.md"
    },
    {
      "source": "work-queue-buildup.md",
      "target": "work-blocking.md"
    },
    {
      "source": "workaround-culture.md",
      "target": "accumulation-of-workarounds.md"
    },
    {
      "source": "workaround-culture.md",
      "target": "increased-technical-shortcuts.md"
    },
    {
      "source": "workaround-culture.md",
      "target": "work-blocking.md"
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