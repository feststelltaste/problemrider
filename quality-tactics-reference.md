# Quality Tactics Reference

All tactics from the [Quality Tactics](https://qualitytactics.de/en/) book, organized by quality characteristic.
This file is a standalone reference so the catalog can be maintained without access to the `qualitaetstaktiken` repo.
It is the data source for `scripts/sync_quality_tactics.py` when the sibling repo is not available.

## Compatibility (54)

| Title | Short Description | URL |
|---|---|---|
| Abstraction | Decouple components through contracts so that implementations can vary independently | https://qualitytactics.de/en/compatibility/abstraction/ |
| Adapter | Translate between incompatible interfaces through an intermediary layer | https://qualitytactics.de/en/compatibility/adapter/ |
| Anti Corruption Layer | Protect existing systems from negative influences of external systems | https://qualitytactics.de/en/compatibility/anti-corruption-layer/ |
| API Deprecation Policy | Retiring old interfaces with sunset headers, timelines, and migration guides | https://qualitytactics.de/en/compatibility/api-deprecation-policy/ |
| API Gateway | Centralizing protocol translation, versioning, and routing through a single entry point | https://qualitytactics.de/en/compatibility/api-gateway/ |
| API Versioning Strategy | Choose a concrete mechanism to identify and route between API versions | https://qualitytactics.de/en/compatibility/api-versioning-strategy/ |
| Backward Compatibility | Guaranteeing that new versions continue to work with existing clients, data, and integrations | https://qualitytactics.de/en/compatibility/backward-compatibility/ |
| Backward Compatible APIs | Evolving API contracts without breaking existing consumers | https://qualitytactics.de/en/compatibility/backward-compatible-apis/ |
| Backward Compatible Data Formats | Ensuring backward compatibility when introducing new data formats | https://qualitytactics.de/en/compatibility/backward-compatible-data-formats/ |
| Backward-Compatible Schema Migrations | Consider backward compatibility in database schemas and migrations | https://qualitytactics.de/en/compatibility/backward-compatible-schema-migrations/ |
| Bridges | Let abstraction hierarchies and implementation hierarchies evolve independently | https://qualitytactics.de/en/compatibility/bridges/ |
| Browser Compatibility | Ensuring browser compatibility through the use of web standards and progressive enhancement | https://qualitytactics.de/en/compatibility/browser-compatibility/ |
| Canonical Data Model | Standardizing a shared data model across systems instead of point-to-point transformations | https://qualitytactics.de/en/compatibility/canonical-data-model/ |
| Compatibility as Error | Treat compatibility regressions as build-breaking defects, not as acceptable technical debt | https://qualitytactics.de/en/compatibility/compatibility-as-error/ |
| Compatibility Certification | Obtain third-party attestation that software meets defined compatibility standards | https://qualitytactics.de/en/compatibility/compatibility-certification/ |
| Compatibility Governance | Assign ownership, track issues, and plan compatibility evolution across releases | https://qualitytactics.de/en/compatibility/compatibility-governance/ |
| Compatibility Matrix | Define supported combinations of configurations | https://qualitytactics.de/en/compatibility/compatibility-matrix/ |
| Compatibility Measurement | Quantify compatibility status through metrics, audits, and risk assessments | https://qualitytactics.de/en/compatibility/compatibility-measurement/ |
| Compatibility Requirements | Make implicit compatibility assumptions explicit and binding | https://qualitytactics.de/en/compatibility/compatibility-requirements/ |
| Compatibility Standards | Define binding rules for compatible development and enforce them in the delivery process | https://qualitytactics.de/en/compatibility/compatibility-standards/ |
| Compatibility Testing | Verify that software works correctly across target platforms, versions, and integration partners | https://qualitytactics.de/en/compatibility/compatibility-testing/ |
| Compatibility Testing by Users | Ensure compatibility through tests conducted by users | https://qualitytactics.de/en/compatibility/compatibility-testing-by-users/ |
| Consumer Driven Contracts | Contracts that define the expectations of interface users | https://qualitytactics.de/en/compatibility/consumer-driven-contracts/ |
| Containerization | Bundle applications with their exact dependency tree into lightweight, reproducible units | https://qualitytactics.de/en/compatibility/containerization/ |
| Content Negotiation | Letting clients and servers agree on format, language, and encoding via HTTP | https://qualitytactics.de/en/compatibility/content-negotiation/ |
| Continuous Integration | Detect compatibility regressions automatically with every code change | https://qualitytactics.de/en/compatibility/continuous-integration/ |
| Cross-Platform Serialization | Use data serializers that are compatible across different systems | https://qualitytactics.de/en/compatibility/cross-platform-serialization/ |
| Cross-Version Testing | Testing the software with different versions | https://qualitytactics.de/en/compatibility/cross-version-testing/ |
| Data Ecosystems | Enable interoperability through shared data platforms, standards, and exchange protocols | https://qualitytactics.de/en/compatibility/data-ecosystems/ |
| Data Format Conversion | Provide mechanisms for converting between different data formats | https://qualitytactics.de/en/compatibility/data-format-conversion/ |
| Data Formats | Use standardized and widely adopted data formats for data exchange | https://qualitytactics.de/en/compatibility/data-formats/ |
| Data Strategy | Define common data standards, formats, and integration patterns across systems | https://qualitytactics.de/en/compatibility/data-strategy/ |
| Dependency Pinning | Locking external dependency versions for reproducible, compatible builds | https://qualitytactics.de/en/compatibility/dependency-pinning/ |
| Compatibility Documentation | Maintain a living record of supported platforms, versions, and known limitations | https://qualitytactics.de/en/compatibility/documentation-of-compatibility-requirements/ |
| Emulation | Reproduce a foreign platform's behavior so existing software runs without modification | https://qualitytactics.de/en/compatibility/emulation/ |
| Event-Driven Integration | Decoupling producers from consumers via asynchronous message broker communication | https://qualitytactics.de/en/compatibility/event-driven-integration/ |
| Facades | Use facades to hide complex subsystems behind a simplified interface | https://qualitytactics.de/en/compatibility/facades/ |
| Feature Detection | Query system capabilities at runtime instead of relying on version numbers | https://qualitytactics.de/en/compatibility/feature-detection/ |
| Forward Compatibility | Ensure compatibility of existing systems with future versions | https://qualitytactics.de/en/compatibility/forward-compatibility/ |
| Idempotent Operations | Design operations so that repeated execution produces the same result as a single execution | https://qualitytactics.de/en/compatibility/idempotent-operations/ |
| Interoperability Tests | Conduct dedicated interoperability tests | https://qualitytactics.de/en/compatibility/interoperability-tests/ |
| Isolated Test Environments | Provide isolated test environments to verify compatibility and interoperability | https://qualitytactics.de/en/compatibility/isolated-test-environments/ |
| Mediator | Decouple direct communication between components | https://qualitytactics.de/en/compatibility/mediator/ |
| Protocol Abstraction | Decoupling communication protocols through abstraction | https://qualitytactics.de/en/compatibility/protocol-abstraction/ |
| Schema Registry | Managing schemas centrally with enforced data contract compatibility across services | https://qualitytactics.de/en/compatibility/schema-registry/ |
| Semantic Versioning | Communicate compatibility intent through structured version numbers | https://qualitytactics.de/en/compatibility/semantic-versioning/ |
| Service Mesh | Managing traffic at infrastructure level with transparent protocol translation, mTLS, and routing | https://qualitytactics.de/en/compatibility/service-mesh/ |
| Simulation Environments | Recreate real systems as a simulated environment | https://qualitytactics.de/en/compatibility/simulation-environments/ |
| Standardized Interfaces | Adopt widely accepted interface styles so that any consumer can integrate without bespoke adapters | https://qualitytactics.de/en/compatibility/standardized-interfaces/ |
| Standardized Protocols | Select transport and messaging protocols with broad ecosystem support | https://qualitytactics.de/en/compatibility/standardized-protocols/ |
| Tolerant Reader Pattern | Ignoring unknown fields and tolerating structural additions on the consumer side | https://qualitytactics.de/en/compatibility/tolerant-reader/ |
| Version Control for Compatibility | Track and manage compatibility-relevant changes across parallel versions | https://qualitytactics.de/en/compatibility/version-control/ |
| Versioning Scheme | Define when and why version numbers change to signal compatibility intent | https://qualitytactics.de/en/compatibility/versioning-scheme/ |
| Virtualization | Isolate applications with their own OS instance to prevent resource and dependency conflicts | https://qualitytactics.de/en/compatibility/virtualization/ |

## Functional Suitability (62)

| Title | Short Description | URL |
|---|---|---|
| Acceptance Tests | Verify fulfillment of business requirements through automated tests | https://qualitytactics.de/en/functional-suitability/acceptance-tests/ |
| Architecture Reviews | Conduct regular evaluations of the software architecture | https://qualitytactics.de/en/functional-suitability/architecture-reviews/ |
| Behavior-Driven Development (BDD) | Formulating requirements as executable scenarios in natural language | https://qualitytactics.de/en/functional-suitability/behavior-driven-development-bdd/ |
| Bounded Contexts | Separate business areas with different terms and rules from each other | https://qualitytactics.de/en/functional-suitability/bounded-contexts/ |
| Business Event Processing | Recognize, process, and respond to business events | https://qualitytactics.de/en/functional-suitability/business-event-processing/ |
| Business Metrics | Define business metrics to evaluate the functionality and quality of the software | https://qualitytactics.de/en/functional-suitability/business-metrics/ |
| Business Process Automation | Mapping business concepts and rules in an executable model | https://qualitytactics.de/en/functional-suitability/business-process-automation/ |
| Business Process Modeling | Elicit business requirements by modeling the underlying business processes | https://qualitytactics.de/en/functional-suitability/business-process-modeling/ |
| Business Quality Scenarios | Specify and verify quality requirements through business-driven scenarios | https://qualitytactics.de/en/functional-suitability/business-quality-scenarios/ |
| Business Test Cases | Create test cases from a business perspective and have them reviewed by users | https://qualitytactics.de/en/functional-suitability/business-test-cases/ |
| Code Reviews | Conduct regular reviews of the source code by team members | https://qualitytactics.de/en/functional-suitability/code-reviews/ |
| Continuous Delivery | Deliver functionality frequently and incrementally | https://qualitytactics.de/en/functional-suitability/continuous-delivery/ |
| Continuous Feedback | Regularly gather and implement feedback from users and stakeholders | https://qualitytactics.de/en/functional-suitability/continuous-feedback/ |
| Customizing | Adapting software to the specific requirements and needs of users | https://qualitytactics.de/en/functional-suitability/customizing/ |
| Data Enrichment | Supplementing data with additional information from external sources | https://qualitytactics.de/en/functional-suitability/data-enrichment/ |
| Data Integration | Merging data from various sources and providing it uniformly | https://qualitytactics.de/en/functional-suitability/data-integration/ |
| Data Modeling | Mapping business concepts and relationships in a conceptual data model | https://qualitytactics.de/en/functional-suitability/data-modeling/ |
| Data Quality Checks | Ensuring data quality through validation, cleansing, and enrichment | https://qualitytactics.de/en/functional-suitability/data-quality-checks/ |
| Datensparsamkeit | Collecting and storing only necessary data | https://qualitytactics.de/en/functional-suitability/datensparsamkeit/ |
| Decision Tables | Define and evaluate complex business rules in tabular form | https://qualitytactics.de/en/functional-suitability/decision-tables/ |
| Definition of Done | Define clear criteria for the completion of functionality | https://qualitytactics.de/en/functional-suitability/definition-of-done/ |
| Design by Contract | Specifying preconditions, postconditions, and invariants for explicit, verifiable behavior | https://qualitytactics.de/en/functional-suitability/design-by-contract/ |
| Direct Feedback | Gather feedback from users directly in the software system | https://qualitytactics.de/en/functional-suitability/direct-feedback/ |
| Domain-Aligned Architecture | Aligning the software's structure with domain structures and processes | https://qualitytactics.de/en/functional-suitability/domain-aligned-architecture/ |
| Domain-Based Authorization Concept | Control access to sensitive data based on business authorizations | https://qualitytactics.de/en/functional-suitability/domain-based-authorization-concept/ |
| Domain Data Versioning | Track and restore changes to domain-specific data | https://qualitytactics.de/en/functional-suitability/domain-data-versioning/ |
| Domain-Driven Design | Align software architecture with the domain's business structures and terms | https://qualitytactics.de/en/functional-suitability/domain-driven-design/ |
| Domain Experts | Directly involve domain experts in development | https://qualitytactics.de/en/functional-suitability/domain-experts/ |
| Domain Modeling | Mapping domain concepts and relationships in a domain model | https://qualitytactics.de/en/functional-suitability/domain-modeling/ |
| Domain Patterns | Applying proven solutions for recurring business problems | https://qualitytactics.de/en/functional-suitability/domain-patterns/ |
| Domain Quiz | Testing domain knowledge through targeted questions | https://qualitytactics.de/en/functional-suitability/domain-quiz/ |
| Domain-Specific Languages | Use programming languages specifically adapted to the domain for business expressions and rules | https://qualitytactics.de/en/functional-suitability/domain-specific-languages/ |
| Event Storming | Discovering domain events, commands, and aggregates in collaborative workshops | https://qualitytactics.de/en/functional-suitability/event-storming/ |
| Evolutionary Requirements Development | Detailing and refining requirements incrementally throughout the project | https://qualitytactics.de/en/functional-suitability/evolutionary-requirements-development/ |
| Feature Driven Development | Structuring and implementing software functionality in the form of features | https://qualitytactics.de/en/functional-suitability/feature-driven-development/ |
| Feature Flags | Toggling feature availability at runtime per user segment | https://qualitytactics.de/en/functional-suitability/feature-flags/ |
| Functional Debt Management | Identify and prioritize problematic implementation of functional requirements | https://qualitytactics.de/en/functional-suitability/functional-debt-management/ |
| Functional Gap Analysis | Identifying missing functionality by comparing capabilities against requirements | https://qualitytactics.de/en/functional-suitability/functional-gap-analysis/ |
| Functional Spike | Investigate business risks through time-limited experiments | https://qualitytactics.de/en/functional-suitability/functional-spike/ |
| Functional Tests | Verify the software's functionality through systematic testing | https://qualitytactics.de/en/functional-suitability/functional-tests/ |
| Impact Mapping | Mapping business goals through actors and impacts to concrete deliverables | https://qualitytactics.de/en/functional-suitability/impact-mapping/ |
| Iterative Development | Develop and deliver software incrementally in short cycles | https://qualitytactics.de/en/functional-suitability/iterative-development/ |
| Microservices | Enabling rapid product experimentation through independent, business-aligned services | https://qualitytactics.de/en/functional-suitability/microservices/ |
| On-Site Customer | Directly involve customers in development | https://qualitytactics.de/en/functional-suitability/on-site-customer/ |
| Personas | Characterizing representative user types through fictional characters | https://qualitytactics.de/en/functional-suitability/personas/ |
| Product Owner | Assign responsibility for business requirements and acceptance to a dedicated role | https://qualitytactics.de/en/functional-suitability/product-owner/ |
| Prototypes | Validate suitability and usability early through business prototypes | https://qualitytactics.de/en/functional-suitability/prototypes/ |
| Prototyping | Gather early feedback on functionality and usability | https://qualitytactics.de/en/functional-suitability/prototyping/ |
| Regression Testing | Re-running existing tests after every change against unintended breakage | https://qualitytactics.de/en/functional-suitability/regression-testing/ |
| Requirements Analysis | Systematic collection, analysis, and documentation of functional requirements | https://qualitytactics.de/en/functional-suitability/requirements-analysis/ |
| Requirements Traceability Matrix | Maintaining explicit bidirectional mappings from requirements through design, code, and tests | https://qualitytactics.de/en/functional-suitability/requirements-traceability-matrix/ |
| Rule-Based Systems | Defining rules that govern the behavior of the software | https://qualitytactics.de/en/functional-suitability/rule-based-systems/ |
| Specification by Example | Collaboratively defining requirements through concrete examples that become executable specifications | https://qualitytactics.de/en/functional-suitability/specification-by-example/ |
| Standard Software | Use proven standard software instead of developing ordinary functionality yourself | https://qualitytactics.de/en/functional-suitability/standard-software/ |
| Story Mapping | Visualizing complete user journeys as a two-dimensional map of gaps and priorities | https://qualitytactics.de/en/functional-suitability/story-mapping/ |
| Subject Matter Reviews | Have work results reviewed and approved by domain experts | https://qualitytactics.de/en/functional-suitability/subject-matter-reviews/ |
| Tracer Bullets | Validate end-to-end functionality early through simplified implementations | https://qualitytactics.de/en/functional-suitability/tracer-bullets/ |
| Ubiquitous Language | Aligning developer and domain expert vocabulary in code and conversation | https://qualitytactics.de/en/functional-suitability/ubiquitous-language/ |
| Usability Tests | Verify usability and suitability through user tests | https://qualitytactics.de/en/functional-suitability/usability-tests/ |
| User Acceptance Tests | Confirm fulfillment of requirements through formal acceptance tests with users | https://qualitytactics.de/en/functional-suitability/user-acceptance-tests/ |
| User Stories | Formulate requirements from the user's perspective | https://qualitytactics.de/en/functional-suitability/user-stories/ |
| Value Range Definition | Define acceptable value ranges for inputs and outputs | https://qualitytactics.de/en/functional-suitability/value-range-definition/ |

## Maintainability (72)

| Title | Short Description | URL |
|---|---|---|
| Short Iteration Cycles | Force incremental, maintainable design through time-boxed delivery cycles | https://qualitytactics.de/en/maintainability/agile-development-methods/ |
| Anti Corruption Layer | Decouple the internal design and logic of a system from external influences | https://qualitytactics.de/en/maintainability/anti-corruption-layer/ |
| API Documentation | Describe interfaces and their usage in detail | https://qualitytactics.de/en/maintainability/api-documentation/ |
| API-First Design | Define and design interfaces before implementing application logic | https://qualitytactics.de/en/maintainability/api-first-design/ |
| Architecture Conformity Analysis | Check the alignment of the software architecture with defined architectural principles | https://qualitytactics.de/en/maintainability/architecture-conformity-analysis/ |
| Architecture Decision Records (ADR) | Documenting important architectural decisions and their justifications | https://qualitytactics.de/en/maintainability/architecture-decision-records-adr/ |
| Architecture Documentation | Create and maintain detailed documentation of the software architecture | https://qualitytactics.de/en/maintainability/architecture-documentation/ |
| Architecture Governance | Definition and enforcement of architectural principles and best practices | https://qualitytactics.de/en/maintainability/architecture-governance/ |
| Architecture Review Board | Establishment of a committee for monitoring and controlling architecture development | https://qualitytactics.de/en/maintainability/architecture-review-board/ |
| Architecture Reviews | Regular systematic review of the software architecture | https://qualitytactics.de/en/maintainability/architecture-reviews/ |
| Architecture Roadmap | Long-term planning and management of architecture development | https://qualitytactics.de/en/maintainability/architecture-roadmap/ |
| Architecture Workshops | Conduct regular workshops to evolve the software architecture | https://qualitytactics.de/en/maintainability/architecture-workshops/ |
| Aspect-Oriented Programming (AOP) | Separate cross-cutting concerns from the main functionality | https://qualitytactics.de/en/maintainability/aspect-oriented-programming-aop/ |
| Automated Tests | Automatically conduct and regularly execute software tests | https://qualitytactics.de/en/maintainability/automated-tests/ |
| Behavior-Driven Development (BDD) | Development based on expected system behaviors | https://qualitytactics.de/en/maintainability/behavior-driven-development-bdd/ |
| Bubble Context | Clearly distinguish extensions from existing code parts | https://qualitytactics.de/en/maintainability/bubble-context/ |
| Clean Code | Structure source code according to established principles for readability and maintainability | https://qualitytactics.de/en/maintainability/clean-code/ |
| Code Comments | Enhance code with meaningful comments and documentation blocks | https://qualitytactics.de/en/maintainability/code-comments/ |
| Code Conventions | Define and enforce uniform guidelines for code formatting and structure | https://qualitytactics.de/en/maintainability/code-conventions/ |
| Code Coverage Analysis | Measurement of the proportion of code covered by tests | https://qualitytactics.de/en/maintainability/code-coverage-analysis/ |
| Code Generation | Automatic creation of code parts based on templates or metadata | https://qualitytactics.de/en/maintainability/code-generation/ |
| Code Metrics | Collecting and analyzing quantitative measures to evaluate code quality | https://qualitytactics.de/en/maintainability/code-metrics/ |
| Code Quality Gates | Ensure code quality through standardized, automated checks | https://qualitytactics.de/en/maintainability/code-quality-gates/ |
| Code Reviews | Systematic review of the source code by other developers | https://qualitytactics.de/en/maintainability/code-reviews/ |
| Ensemble Programming | Solve complex design and debugging challenges by programming as a group at one workstation | https://qualitytactics.de/en/maintainability/collaborative-problem-solving/ |
| Containerization | Encapsulating applications and their dependencies in containers | https://qualitytactics.de/en/maintainability/containerization/ |
| Continuous Delivery | Automated preparation of software changes for the production environment | https://qualitytactics.de/en/maintainability/continuous-delivery/ |
| Continuous Deployment | Fully automated deployment of software changes in the production environment | https://qualitytactics.de/en/maintainability/continuous-deployment/ |
| Continuous Integration | Regular integration of code changes into a shared repository | https://qualitytactics.de/en/maintainability/continuous-integration/ |
| Contract Testing | Verifying service interfaces conform to agreed contracts for independent modification | https://qualitytactics.de/en/maintainability/contract-testing/ |
| Dependency Injection | Manage and inject dependencies between components externally | https://qualitytactics.de/en/maintainability/dependency-injection/ |
| Dependency Injection Container | Centralized management and provision of dependencies | https://qualitytactics.de/en/maintainability/dependency-injection-container/ |
| Dependency Management | Systematize the management and updating of external dependencies | https://qualitytactics.de/en/maintainability/dependency-management/ |
| Deprecation Strategy | Systematically mark and gradually remove deprecated features | https://qualitytactics.de/en/maintainability/deprecation-strategy/ |
| Docs as Code | Treat and manage documentation like source code | https://qualitytactics.de/en/maintainability/docs-as-code/ |
| Domain-Driven Design | Structuring software architecture based on the business domain | https://qualitytactics.de/en/maintainability/domain-driven-design/ |
| Event-Driven Architecture | Decoupling components through asynchronous events for independent evolution and modification | https://qualitytactics.de/en/maintainability/event-driven-architecture/ |
| Evolutionary Database Design | Evolving database schemas incrementally through version-controlled migration scripts | https://qualitytactics.de/en/maintainability/evolutionary-database-design/ |
| Open Development Practices | Improve code quality through public code review, transparent issue tracking, and external contributions | https://qualitytactics.de/en/maintainability/fair-source/ |
| Feature Toggles | Enable or disable functions through configuration switches | https://qualitytactics.de/en/maintainability/feature-toggles/ |
| Fitness Functions | Regular review of compliance with architectural guidelines | https://qualitytactics.de/en/maintainability/fitness-functions/ |
| Fluent Interfaces | API design with natural language-like method chaining | https://qualitytactics.de/en/maintainability/fluent-interfaces/ |
| Hexagonal Architecture | Isolating business logic from infrastructure through ports and adapters | https://qualitytactics.de/en/maintainability/hexagonal-architecture/ |
| High Cohesion | Ensuring each module has a focused, well-defined purpose with closely related responsibilities | https://qualitytactics.de/en/maintainability/high-cohesion/ |
| Infrastructure as Code | Defining and managing infrastructure through code | https://qualitytactics.de/en/maintainability/infrastructure-as-code/ |
| Integration Tests | Conduct tests to verify the interaction of different system components | https://qualitytactics.de/en/maintainability/integration-tests/ |
| Knowledge Management System | Collect and distribute knowledge about the software project centrally | https://qualitytactics.de/en/maintainability/knowledge-management-system/ |
| Layered Architecture | Divide software system into logical layers with clear responsibilities | https://qualitytactics.de/en/maintainability/layered-architecture/ |
| Living Documentation | Current and easily accessible documentation as an integral part of development | https://qualitytactics.de/en/maintainability/living-documentation/ |
| Logging | Implement comprehensive logging and monitoring of system behavior | https://qualitytactics.de/en/maintainability/logging/ |
| Loose Coupling | Minimizing dependencies between modules so changes in one don't cascade | https://qualitytactics.de/en/maintainability/loose-coupling/ |
| Microservices | Division of the application into small, independent services | https://qualitytactics.de/en/maintainability/microservices/ |
| Modularization | Divide application into small, independent, and reusable components | https://qualitytactics.de/en/maintainability/modularization/ |
| Modulith | Structure system architecture into independent, interchangeable modules | https://qualitytactics.de/en/maintainability/modulith/ |
| Mutation Testing | Testing the robustness of software tests through targeted code changes | https://qualitytactics.de/en/maintainability/mutation-testing/ |
| Observability | Implementing structured logging, distributed tracing, and metrics for deep system understanding | https://qualitytactics.de/en/maintainability/observability/ |
| Pair Programming | Two developers work together on a task at one workstation | https://qualitytactics.de/en/maintainability/pair-programming/ |
| Pattern Language | Apply proven solution patterns for recurring design problems | https://qualitytactics.de/en/maintainability/pattern-language/ |
| Property-Based Testing | Verify software through random inputs and properties | https://qualitytactics.de/en/maintainability/property-based-testing/ |
| Refactoring | Regular revision of the code to improve the internal structure | https://qualitytactics.de/en/maintainability/refactoring/ |
| Refactoring Katas | Perform regular exercises to improve code quality | https://qualitytactics.de/en/maintainability/refactoring-katas/ |
| Separation of Concerns | Divide functionalities into clearly defined, independent areas | https://qualitytactics.de/en/maintainability/separation-of-concerns/ |
| SOLID Principles | Apply fundamental design principles for object-oriented programming | https://qualitytactics.de/en/maintainability/solid-principles/ |
| Static Code Analysis | Automated review of source code for potential issues and improvement opportunities | https://qualitytactics.de/en/maintainability/static-code-analysis/ |
| Strangler Fig Pattern | Replacing legacy systems incrementally by routing traffic to new implementations | https://qualitytactics.de/en/maintainability/strangler-fig-pattern/ |
| Strategic Code Deletion | Targeted removal of superfluous or obsolete code to reduce the codebase | https://qualitytactics.de/en/maintainability/strategic-code-deletion/ |
| Technical Debt Management | Identifying, tracking, and prioritizing technical debt for long-term modifiability | https://qualitytactics.de/en/maintainability/technical-debt-management/ |
| Technical Spike | Validate that an architecture will remain maintainable under expected growth | https://qualitytactics.de/en/maintainability/technical-spike/ |
| Test-Driven Development (TDD) | Writing tests before the actual implementation | https://qualitytactics.de/en/maintainability/test-driven-development-tdd/ |
| Trunk-Based Development | Integrating short-lived branches continuously into main for rapid, safe modifications | https://qualitytactics.de/en/maintainability/trunk-based-development/ |
| Version Control | Systematically track and manage changes to the source code | https://qualitytactics.de/en/maintainability/version-control/ |
| Walking Skeleton | Develop a minimal, running system with the core architectural ideas | https://qualitytactics.de/en/maintainability/walking-skeleton/ |

## Performance Efficiency (76)

| Title | Short Description | URL |
|---|---|---|
| Adaptive Streaming | Adapt data streams dynamically to the network conditions | https://qualitytactics.de/en/performance-efficiency/adaptive-streaming/ |
| API Calls Optimization | Designing API calls efficiently | https://qualitytactics.de/en/performance-efficiency/api-calls-optimization/ |
| Approximation Methods | Use of heuristics and estimations instead of exact calculations | https://qualitytactics.de/en/performance-efficiency/approximation-methods/ |
| Asynchronous Logging | Decoupling the logging process from the main application | https://qualitytactics.de/en/performance-efficiency/asynchronous-logging/ |
| Asynchronous Processing | Decoupling of calls and execution through asynchronicity | https://qualitytactics.de/en/performance-efficiency/asynchronous-processing/ |
| Backpressure | Signaling producers to slow down when consumers become overwhelmed | https://qualitytactics.de/en/performance-efficiency/backpressure/ |
| Batch Processing | Collecting and processing multiple jobs together | https://qualitytactics.de/en/performance-efficiency/batch-processing/ |
| Caching | Caching frequently needed data | https://qualitytactics.de/en/performance-efficiency/caching/ |
| Capacity Planning | Estimating future resource needs from growth projections and performance models | https://qualitytactics.de/en/performance-efficiency/capacity-planning/ |
| Client Side Rendering | Shifting the rendering process of web pages from the server to the client | https://qualitytactics.de/en/performance-efficiency/client-side-rendering/ |
| Code Splitting | Splitting the application code into smaller chunks | https://qualitytactics.de/en/performance-efficiency/code-splitting/ |
| Cold Start Mitigation | Reducing initialization latency in serverless, container, and JVM applications proactively | https://qualitytactics.de/en/performance-efficiency/cold-start-mitigation/ |
| Compression | Reduce storage space with or without loss | https://qualitytactics.de/en/performance-efficiency/compression/ |
| Concurrency | Simultaneous execution of multiple tasks within a process | https://qualitytactics.de/en/performance-efficiency/concurrency/ |
| Connection Pooling | Reusing pre-established connections instead of creating new ones per request | https://qualitytactics.de/en/performance-efficiency/connection-pooling/ |
| Content Delivery Networks | Distribute content geographically across a network of servers | https://qualitytactics.de/en/performance-efficiency/content-delivery-networks/ |
| Continuous Performance Monitoring | Ongoing monitoring and analysis of application performance in production | https://qualitytactics.de/en/performance-efficiency/continuous-performance-monitoring/ |
| CQRS | Separating read and write models into independently optimized and scaled paths | https://qualitytactics.de/en/performance-efficiency/cqrs/ |
| Data Aggregation | Summarize fine-grained data into more compact units | https://qualitytactics.de/en/performance-efficiency/data-aggregation/ |
| Data Archiving | Offloading infrequently needed data to more cost-effective storage media | https://qualitytactics.de/en/performance-efficiency/data-archiving/ |
| Data Deduplication | Detection and elimination of redundant data in storage systems | https://qualitytactics.de/en/performance-efficiency/data-deduplication/ |
| Data Partitioning | Division of large datasets across multiple computers or storage units | https://qualitytactics.de/en/performance-efficiency/data-partitioning/ |
| Data Stream Processing | Continuous processing of data from real-time data sources | https://qualitytactics.de/en/performance-efficiency/data-stream-processing/ |
| Database Optimization | Adjustment of database design and configuration for optimal performance | https://qualitytactics.de/en/performance-efficiency/database-optimization/ |
| Deferred Static Generation | Generate static web pages only on first request | https://qualitytactics.de/en/performance-efficiency/deferred-static-generation/ |
| Denormalization | Introducing controlled redundancy in database schemas for faster reads | https://qualitytactics.de/en/performance-efficiency/denormalization/ |
| Distributed Caching | Caching frequently needed data on multiple computers | https://qualitytactics.de/en/performance-efficiency/distributed-caching/ |
| Distributed Processing | Division of processing across multiple independent systems | https://qualitytactics.de/en/performance-efficiency/distributed-processing/ |
| Distributed Tracing | Tracking requests across microservice boundaries with their performance impact | https://qualitytactics.de/en/performance-efficiency/distributed-tracing/ |
| Dynamic Programming | Decomposition of a problem into overlapping subproblems and storage of intermediate results | https://qualitytactics.de/en/performance-efficiency/dynamic-programming/ |
| Edge Computing | Perform data processing closer to the source | https://qualitytactics.de/en/performance-efficiency/edge-computing/ |
| Efficient Algorithms | Choosing efficient algorithms for frequent or critical operations | https://qualitytactics.de/en/performance-efficiency/efficient-algorithms/ |
| Elastic Scaling | Dynamic adjustment of resource allocation to the current load | https://qualitytactics.de/en/performance-efficiency/elastic-scaling/ |
| Graph Databases | Enable the storage and querying of connected data in the form of nodes and edges | https://qualitytactics.de/en/performance-efficiency/graph-databases/ |
| Horizontal Scaling | Increasing performance by adding additional components | https://qualitytactics.de/en/performance-efficiency/horizontal-scaling/ |
| Image and Asset Optimization | Optimizing images, fonts, and static assets for smaller payloads and faster loads | https://qualitytactics.de/en/performance-efficiency/image-and-asset-optimization/ |
| In-Memory Processing | Keeping all data in main memory instead of on slow storage media | https://qualitytactics.de/en/performance-efficiency/in-memory-processing/ |
| Infinite Scrolling | Reduce initial payload and server load through incremental, on-demand data fetching | https://qualitytactics.de/en/performance-efficiency/infinite-scrolling/ |
| Lazy Evaluation | Load and process data only when needed | https://qualitytactics.de/en/performance-efficiency/lazy-evaluation/ |
| Lazy Loading | Delayed loading of data and resources until the moment of actual use | https://qualitytactics.de/en/performance-efficiency/lazy-loading/ |
| Load Balancing | Distribution of the load across multiple parallel processing units | https://qualitytactics.de/en/performance-efficiency/load-balancing/ |
| Load Testing | Testing the software under high load | https://qualitytactics.de/en/performance-efficiency/load-testing/ |
| Mass Test Data Generation | Generation of massive artificial test data with realistic properties | https://qualitytactics.de/en/performance-efficiency/mass-test-data-generation/ |
| Materialized Views | Optimize database query performance by storing query results | https://qualitytactics.de/en/performance-efficiency/materialized-views/ |
| Memory Hierarchy | Utilizing locality of memory accesses at different levels | https://qualitytactics.de/en/performance-efficiency/memory-hierarchy/ |
| NoSQL Databases | Storing data in flexible, schema-less formats | https://qualitytactics.de/en/performance-efficiency/nosql-databases/ |
| Optimistic UI Updates | Reduce perceived latency by updating the interface before server confirmation | https://qualitytactics.de/en/performance-efficiency/optimistic-ui-updates/ |
| Pagination | Loading large outputs of data into smaller, manageable chunks | https://qualitytactics.de/en/performance-efficiency/pagination/ |
| Parallelization | Simultaneous execution of multiple calculations or tasks | https://qualitytactics.de/en/performance-efficiency/parallelization/ |
| Performance Budgets | Defining performance indicators as part of the requirements | https://qualitytactics.de/en/performance-efficiency/performance-budgets/ |
| Performance Measurements | Continuous measurement and storage of performance metrics in production | https://qualitytactics.de/en/performance-efficiency/performance-measurements/ |
| Performance Modeling | Predicting performance behavior through mathematical models | https://qualitytactics.de/en/performance-efficiency/performance-modeling/ |
| Pipelining | Simultaneous execution of sequential processing steps | https://qualitytactics.de/en/performance-efficiency/pipelining/ |
| Predictive Loading | Proactive loading of data likely to be needed next | https://qualitytactics.de/en/performance-efficiency/predictive-loading/ |
| Predictive Prefetching | Loading of probably required content derived from current usage | https://qualitytactics.de/en/performance-efficiency/predictive-prefetching/ |
| Probabilistic Data Structures | Using data structures that trade accuracy for space | https://qualitytactics.de/en/performance-efficiency/probabilistic-data-structures/ |
| Profiling | Analyzing applications regarding their performance in detail | https://qualitytactics.de/en/performance-efficiency/profiling/ |
| Progressive Loading | Incremental loading of content with increasing quality | https://qualitytactics.de/en/performance-efficiency/progressive-loading/ |
| Reactive Programming | Development of applications that react to events and process data flows | https://qualitytactics.de/en/performance-efficiency/reactive-programming/ |
| Read Replicas | Distributing query load across read-only database replicas away from the primary | https://qualitytactics.de/en/performance-efficiency/read-replicas/ |
| Resource Pooling | Shared use of resources by aggregating into pools | https://qualitytactics.de/en/performance-efficiency/resource-pooling/ |
| Resource Usage Optimization | Minimization of the consumption of scarce resources | https://qualitytactics.de/en/performance-efficiency/resource-usage-optimization/ |
| Sampling | Using a representative subset of data for analysis or testing | https://qualitytactics.de/en/performance-efficiency/sampling/ |
| Serialization Optimization | Choosing efficient serialization formats for performance-critical data exchange | https://qualitytactics.de/en/performance-efficiency/serialization-optimization/ |
| Server Side Rendering | Improves load times and SEO by rendering page content on the server | https://qualitytactics.de/en/performance-efficiency/server-side-rendering/ |
| Serverless Computing | Execution of code without managing the underlying infrastructure | https://qualitytactics.de/en/performance-efficiency/serverless-computing/ |
| Specialized Hardware | Use of hardware-accelerated functions or specialized hardware components | https://qualitytactics.de/en/performance-efficiency/specialized-hardware/ |
| Static Code Analysis | Automated review of source code for performance issues | https://qualitytactics.de/en/performance-efficiency/static-code-analysis/ |
| Static Site Generation | Generate static HTML files at build time | https://qualitytactics.de/en/performance-efficiency/static-site-generation/ |
| Streaming | Continuous processing and transmission of data | https://qualitytactics.de/en/performance-efficiency/streaming/ |
| Stress Testing | Testing the software under extreme load conditions | https://qualitytactics.de/en/performance-efficiency/stress-testing/ |
| Transparent Performance Metrics | Open presentation of system performance and processing times | https://qualitytactics.de/en/performance-efficiency/transparent-performance-metrics/ |
| Tree Shaking | Eliminating unused code while building | https://qualitytactics.de/en/performance-efficiency/tree-shaking/ |
| Vectorization | Utilization of special instructions of modern processors | https://qualitytactics.de/en/performance-efficiency/vectorization/ |
| Vertical Scaling | Increasing the performance of individual components | https://qualitytactics.de/en/performance-efficiency/vertical-scaling/ |
| Virtualized Lists | Efficient display of large data lists through virtual scroll areas | https://qualitytactics.de/en/performance-efficiency/virtualized-lists/ |

## Portability (44)

| Title | Short Description | URL |
|---|---|---|
| Abstracted File System Access | Implementing file system operations through an abstraction layer | https://qualitytactics.de/en/portability/abstracted-file-system-access/ |
| Abstraction Layers | Encapsulating hardware-specific details through abstraction layers | https://qualitytactics.de/en/portability/abstraction-layers/ |
| API-First Development | Developing applications with clearly defined APIs as the foundation | https://qualitytactics.de/en/portability/api-first-development/ |
| Automated Migration Tools | Automating data, configuration, and state migration when transferring between environments | https://qualitytactics.de/en/portability/automated-migration-tools/ |
| Clean Uninstallation | Ensuring software can be cleanly removed without leaving artifacts, registry entries, or orphaned data | https://qualitytactics.de/en/portability/clean-uninstallation/ |
| Cloud-Native Development | Developing and optimizing applications specifically for cloud environments | https://qualitytactics.de/en/portability/cloud-native-development/ |
| Containerization | Packaging applications and their dependencies into isolated containers | https://qualitytactics.de/en/portability/containerization/ |
| Containerized Databases | Deploying databases in containers | https://qualitytactics.de/en/portability/containerized-databases/ |
| Cross-Platform Build Scripts | Implementing build processes with cross-platform scripting languages | https://qualitytactics.de/en/portability/cross-platform-build-scripts/ |
| Cross-Platform Build Tools | Use build tools that can compile for multiple platforms | https://qualitytactics.de/en/portability/cross-platform-build-tools/ |
| Cross-Platform Encryption Libraries | Use encryption libraries that function identically across different systems | https://qualitytactics.de/en/portability/cross-platform-encryption-libraries/ |
| Cross-Platform Frameworks | Utilize development frameworks that enable cross-platform applications | https://qualitytactics.de/en/portability/cross-platform-frameworks/ |
| Cross-Platform Graphics Libraries | Using graphics libraries that render consistently across different systems | https://qualitytactics.de/en/portability/cross-platform-graphics-libraries/ |
| Cross-Platform Package Managers | Using package managers that work on different operating systems | https://qualitytactics.de/en/portability/cross-platform-package-managers/ |
| Cross-Platform UI Frameworks | Utilize UI frameworks that function consistently across different platforms | https://qualitytactics.de/en/portability/cross-platform-ui-frameworks/ |
| Data Export and Liberation | Enabling users to export their data in standard portable formats for migration and compliance | https://qualitytactics.de/en/portability/data-export/ |
| Database Abstraction | Implementing database accesses through an abstracted layer | https://qualitytactics.de/en/portability/database-abstraction/ |
| Dependency Injection | Manage dependencies externally and inject them at runtime | https://qualitytactics.de/en/portability/dependency-injection/ |
| Environment Variables for Configuration | Control configuration settings via environment variables | https://qualitytactics.de/en/portability/environment-variables-for-configuration/ |
| Externalized Configuration | Separate environment-specific settings and application logic | https://qualitytactics.de/en/portability/externalized-configuration/ |
| Feature Detection | Probing platform capabilities at runtime with fallbacks instead of conditional compilation | https://qualitytactics.de/en/portability/feature-detection/ |
| Microservices Architecture | Divide application into small, independent services | https://qualitytactics.de/en/portability/microservices-architecture/ |
| Multi-Architecture Container Builds | Building container images for multiple CPU architectures using tools like Docker Buildx | https://qualitytactics.de/en/portability/multi-architecture-container-builds/ |
| Multi-Cloud Infrastructure as Code | Provisioning infrastructure declaratively with provider-agnostic modules for multiple clouds | https://qualitytactics.de/en/portability/multi-cloud-iac/ |
| Object-Relational Mapping (ORM) | Abstracting database interactions through objects | https://qualitytactics.de/en/portability/object-relational-mapping-orm/ |
| Platform Independence | Make software executable on different systems and environments without modifications | https://qualitytactics.de/en/portability/platform-independence/ |
| Platform-Independent Build Pipelines | Implementing CI/CD pipelines that run on different build servers | https://qualitytactics.de/en/portability/platform-independent-build-pipelines/ |
| Platform-Independent Configuration Files | Store configurations in standardized, platform-independent formats | https://qualitytactics.de/en/portability/platform-independent-configuration-files/ |
| Platform-Independent Configuration Management | Store configuration settings in platform-independent formats | https://qualitytactics.de/en/portability/platform-independent-configuration-management/ |
| Platform-Independent Data Storage | Choose database systems and storage solutions that are available on various platforms | https://qualitytactics.de/en/portability/platform-independent-data-storage/ |
| Platform-Independent Logging Frameworks | Using logging frameworks that function consistently across different systems | https://qualitytactics.de/en/portability/platform-independent-logging-frameworks/ |
| Platform-Independent Programming Languages | Using programming languages that run on different systems without modifications | https://qualitytactics.de/en/portability/platform-independent-programming-languages/ |
| Platform-Independent Scripting Languages | Using scripting languages for automation and configuration | https://qualitytactics.de/en/portability/platform-independent-scripting-languages/ |
| Platform-Independent Test Frameworks | Using test frameworks that function consistently across different platforms | https://qualitytactics.de/en/portability/platform-independent-test-frameworks/ |
| Platform-Independent Time Zone Handling | Manage time zones and date formats through an abstracted layer | https://qualitytactics.de/en/portability/platform-independent-time-zone-handling/ |
| Portability Checklists | Create checklists to check portability with different systems and platforms | https://qualitytactics.de/en/portability/portability-checklists/ |
| Portable Binary Formats | Creating executable files in platform-independent formats | https://qualitytactics.de/en/portability/portable-binary-formats/ |
| Progressive Web App | Deploy one codebase as an installable app across all platforms and devices | https://qualitytactics.de/en/portability/progressive-web-app/ |
| Standardized Data Formats | Use widely adopted, platform-independent data formats for data exchange | https://qualitytactics.de/en/portability/standardized-data-formats/ |
| Standardized Deployment Scripts | Create unified scripts for deployment and configuration across different platforms | https://qualitytactics.de/en/portability/standardized-deployment-scripts/ |
| Virtual Development Environments | Providing development environments in virtual machines or containers | https://qualitytactics.de/en/portability/virtual-development-environments/ |
| Virtual Networks | Abstracting network configurations through virtual networks | https://qualitytactics.de/en/portability/virtual-networks/ |
| Virtualization | Running applications in virtual machines | https://qualitytactics.de/en/portability/virtualization/ |
| WebAssembly Portability | Using WebAssembly as a portable compilation target across browsers, edge runtimes, and server environments | https://qualitytactics.de/en/portability/webassembly-portability/ |

## Quality Illusions (21)

| Title | Short Description | URL |
|---|---|---|
| Artificial Delays | Intentionally introducing short wait times to increase perceived quality | https://qualitytactics.de/en/quality-illusions/artificial-delays/ |
| Artificial Learning Curve | Intentional complication of simple functions to suggest depth | https://qualitytactics.de/en/quality-illusions/artificial-learning-curve/ |
| Artificial Scarcity Indicators | Manufacturing false urgency and scarcity signals pressuring immediate action | https://qualitytactics.de/en/quality-illusions/artificial-scarcity-indicators/ |
| Compliance Theater | Implementing compliance features that create an appearance of regulatory adherence without substantive implementation | https://qualitytactics.de/en/quality-illusions/compliance-theater/ |
| Confirmshaming | Guilting users into system-preferred choices through manipulative UI copy | https://qualitytactics.de/en/quality-illusions/dark-pattern-confirmshaming/ |
| Fake Datensparsamkeit | Pretending minimal data collection while actually gathering extensive data | https://qualitytactics.de/en/quality-illusions/fake-datensparsamkeit/ |
| Fake Localization | Pretending comprehensive internationalization through superficial translations | https://qualitytactics.de/en/quality-illusions/fake-localization/ |
| Fake Progress Bar | Display artificial progress indicators for indefinitely long processes | https://qualitytactics.de/en/quality-illusions/fake-progress-bar/ |
| Fake Social Proof | Fabricating social signals creating an illusion of popularity and trustworthiness | https://qualitytactics.de/en/quality-illusions/fake-social-proof/ |
| Navigation Maze | Intentionally complicating user guidance | https://qualitytactics.de/en/quality-illusions/navigation-maze/ |
| Phantom Functionality | Implementation of placeholder functions without actual functionality | https://qualitytactics.de/en/quality-illusions/phantom-functionality/ |
| Phantom Notifications | Generation of artificial notifications to increase user engagement | https://qualitytactics.de/en/quality-illusions/phantom-notifications/ |
| Placebo Security | Implementing visible but ineffective security measures that create a false sense of protection | https://qualitytactics.de/en/quality-illusions/placebo-security/ |
| Pseudo-AI Interactions | Feigning intelligent application through pre-programmed responses | https://qualitytactics.de/en/quality-illusions/pseudo-ai-interactions/ |
| Pseudo-Multitasking | Representing sequential processes as parallel operations | https://qualitytactics.de/en/quality-illusions/pseudo-multitasking/ |
| Pseudo-Personalization | Faking personalized content through generic algorithms | https://qualitytactics.de/en/quality-illusions/pseudo-personalization/ |
| Shimmer Effect | Animated placeholders for content not yet loaded | https://qualitytactics.de/en/quality-illusions/shimmer-effect/ |
| Simulated Real-Time Data | Display of precomputed data as live updates | https://qualitytactics.de/en/quality-illusions/simulated-real-time-data/ |
| Skeleton Screens | Displaying placeholder layouts during loading | https://qualitytactics.de/en/quality-illusions/skeleton-screens/ |
| Vanity Metrics Dashboard | Displaying impressive-looking but meaningless metrics that don't reflect actual system health or business value | https://qualitytactics.de/en/quality-illusions/vanity-metrics-dashboard/ |
| Wizard of Oz Backend | Performing tasks manually behind the scenes that users believe are automated | https://qualitytactics.de/en/quality-illusions/wizard-of-oz-backend/ |

## Reliability (79)

| Title | Short Description | URL |
|---|---|---|
| Automated Tests | Automated verification of functionality at various levels | https://qualitytactics.de/en/reliability/automated-tests/ |
| Blameless Postmortems | Learning from incidents systematically, focusing on systemic improvements over individual blame | https://qualitytactics.de/en/reliability/blameless-postmortems/ |
| Blue-Green Deployment | Parallel operation of two production environments to minimize downtime | https://qualitytactics.de/en/reliability/blue-green-deployment/ |
| Boring Technologies | Use proven and mature technologies | https://qualitytactics.de/en/reliability/boring-technologies/ |
| Bulkhead | Dividing a system into isolated areas to limit fault propagation | https://qualitytactics.de/en/reliability/bulkhead/ |
| Canary Releases | Gradual introduction of changes for a limited user group to minimize risk | https://qualitytactics.de/en/reliability/canary-releases/ |
| Chaos Engineering | Intentional introduction of disruptions to test system resilience | https://qualitytactics.de/en/reliability/chaos-engineering/ |
| Checklists | Systematically processing steps and requirements | https://qualitytactics.de/en/reliability/checklists/ |
| Checksums | Checksum calculation for detecting data errors or changes | https://qualitytactics.de/en/reliability/checksums/ |
| Circuit Breaker | Mechanism for error and overload protection in distributed systems | https://qualitytactics.de/en/reliability/circuit-breaker/ |
| Continuous Data Verification | Regular verification of data integrity during storage or transmission | https://qualitytactics.de/en/reliability/continuous-data-verification/ |
| Continuous Integration and Delivery | Automated processes for software integration, testing, and deployment | https://qualitytactics.de/en/reliability/continuous-integration-and-delivery/ |
| Dark Launches | Limit blast radius of new features by deploying them hidden to a subset of users | https://qualitytactics.de/en/reliability/dark-launches/ |
| Data Integrity | Mechanisms to ensure data accuracy, consistency, and reliability | https://qualitytactics.de/en/reliability/data-integrity/ |
| Data Replication | Creating and synchronizing copies of data across multiple systems | https://qualitytactics.de/en/reliability/data-replication/ |
| Dead Letter Queue | Routing failed messages to a dedicated queue for later inspection and reprocessing instead of losing them | https://qualitytactics.de/en/reliability/dead-letter-queue/ |
| Disaster Recovery | Methods for restoring operations after disasters or major disruptions | https://qualitytactics.de/en/reliability/disaster-recovery/ |
| Elastic Resource Utilization | Automatic adjustment of resources based on current load | https://qualitytactics.de/en/reliability/elastic-resource-utilization/ |
| Environment Parity | Ensuring consistency between development, test, and production environments | https://qualitytactics.de/en/reliability/environment-parity/ |
| Error Budgets | Quantifying acceptable unreliability as balance between feature velocity and reliability | https://qualitytactics.de/en/reliability/error-budgets/ |
| Error Correction Codes | Using codes to detect and correct errors in data | https://qualitytactics.de/en/reliability/error-correction-codes/ |
| Error Handling | Mechanisms for detecting, logging, and handling errors | https://qualitytactics.de/en/reliability/error-handling/ |
| Error Logging | Capturing and storing errors and exceptions | https://qualitytactics.de/en/reliability/error-logging/ |
| Error Logs | Perform systematic analysis of error logs | https://qualitytactics.de/en/reliability/error-logs/ |
| Error Reporting and Analysis | Systematic capture, analysis, and resolution of errors and issues | https://qualitytactics.de/en/reliability/error-reporting-and-analysis/ |
| Exceptions | Using exceptions for signaling and handling error states | https://qualitytactics.de/en/reliability/exceptions/ |
| Failover Cluster | Maintaining servers or systems as a functional group redundantly | https://qualitytactics.de/en/reliability/failover-cluster/ |
| Failover Mechanisms | Automatic switch to redundant components in case of failure | https://qualitytactics.de/en/reliability/failover-mechanisms/ |
| Fault Containment | Limiting the impact of faults to a small part of the system | https://qualitytactics.de/en/reliability/fault-containment/ |
| Fault-Tolerant Data Structures | Use of data structures that remain operational despite errors or inconsistencies | https://qualitytactics.de/en/reliability/fault-tolerant-data-structures/ |
| Feature Toggles | Activating and deactivating features for flexible rollouts | https://qualitytactics.de/en/reliability/feature-toggles/ |
| Graceful Degradation | Ability of a system to operate in a limited capacity during failures or overload | https://qualitytactics.de/en/reliability/graceful-degradation/ |
| Health Check Endpoints | Exposing standardized health check APIs for load balancer and orchestrator monitoring | https://qualitytactics.de/en/reliability/health-check-endpoints/ |
| Heartbeat | Regular transmission of a component's heartbeat to a monitoring instance | https://qualitytactics.de/en/reliability/heartbeat/ |
| High Availability Architectures | Architectures designed for maximum availability and fault tolerance | https://qualitytactics.de/en/reliability/high-availability-architectures/ |
| Idempotency Design | Designing safely retryable operations without unintended side effects | https://qualitytactics.de/en/reliability/idempotency-design/ |
| Immutable Infrastructure | Not modifying infrastructure components, but replacing them with new versions | https://qualitytactics.de/en/reliability/immutable-infrastructure/ |
| Incident Management | Structured process for handling disruptions and failures | https://qualitytactics.de/en/reliability/incident-management/ |
| Isolation of Faulty Components | Develop mechanisms to isolate faulty components | https://qualitytactics.de/en/reliability/isolation-of-faulty-components/ |
| Load Balancing | Distributing workload across multiple resources | https://qualitytactics.de/en/reliability/load-balancing/ |
| Load Shedding | Deliberately dropping low-priority requests under overload, preserving critical capacity | https://qualitytactics.de/en/reliability/load-shedding/ |
| Load Testing | Evaluating system performance and stability under high load | https://qualitytactics.de/en/reliability/load-testing/ |
| Monitoring | Continuous monitoring of system states, performance, and errors | https://qualitytactics.de/en/reliability/monitoring/ |
| Monitoring System Integrity | Continuous verification of the integrity of system components, configurations, and data | https://qualitytactics.de/en/reliability/monitoring-system-integrity/ |
| Monitoring System Utilization | Continuous monitoring of resource usage and system performance | https://qualitytactics.de/en/reliability/monitoring-system-utilization/ |
| Nonstop Forwarding | Continuous request forwarding despite failures or errors | https://qualitytactics.de/en/reliability/nonstop-forwarding/ |
| On-Call Duty | Ensuring employees are available to quickly respond to incidents and issues | https://qualitytactics.de/en/reliability/on-call-duty/ |
| Ping | Actively sending requests to a component to check its availability | https://qualitytactics.de/en/reliability/ping/ |
| Plausibility Checks | Checking inputs, data, or states for validity to detect potential errors early | https://qualitytactics.de/en/reliability/plausibility-checks/ |
| Proactive Capacity Management | Forecasting and planning required resources based on growth predictions | https://qualitytactics.de/en/reliability/proactive-capacity-management/ |
| Production Environment Maintenance | Conducting regular inspections and maintenance to maintain reliability | https://qualitytactics.de/en/reliability/production-environment-maintenance/ |
| Rate Limiting | Controlling incoming request rates against system overload during traffic spikes | https://qualitytactics.de/en/reliability/rate-limiting/ |
| Redundancy | Multiple instances of critical components or systems | https://qualitytactics.de/en/reliability/redundancy/ |
| Redundant Checksums | Using multiple different checksum algorithms | https://qualitytactics.de/en/reliability/redundant-checksums/ |
| Redundant Data Storage | Storing data on multiple media or systems | https://qualitytactics.de/en/reliability/redundant-data-storage/ |
| Regular Backups | Regular backup of data and system states | https://qualitytactics.de/en/reliability/regular-backups/ |
| Regular Maintenance and Updates | Performing scheduled maintenance and installing updates | https://qualitytactics.de/en/reliability/regular-maintenance-and-updates/ |
| Resilience | Ability of a system to remain operational under adverse conditions or faults | https://qualitytactics.de/en/reliability/resilience/ |
| Restore Points | Regularly back up the system state | https://qualitytactics.de/en/reliability/restore-points/ |
| Retry | Retrying failed operations to handle transient errors | https://qualitytactics.de/en/reliability/retry/ |
| Rollback Mechanisms | Ability to revert changes and return to a previous stable state | https://qualitytactics.de/en/reliability/rollback-mechanisms/ |
| Rolling Updates | Stepwise updating of servers or instances | https://qualitytactics.de/en/reliability/rolling-updates/ |
| Root Cause Analysis | Systematically analyze the causes of failures | https://qualitytactics.de/en/reliability/root-cause-analysis/ |
| Runbooks | Providing detailed instructions for processing tasks and incidents | https://qualitytactics.de/en/reliability/runbooks/ |
| Saga Pattern | Managing distributed transactions through sequences of local transactions with compensating actions | https://qualitytactics.de/en/reliability/saga-pattern/ |
| Secure Software | Prevent reliability incidents caused by security vulnerabilities | https://qualitytactics.de/en/reliability/secure-software/ |
| Self-Monitoring and Diagnosis | A system's ability to monitor its own state and detect issues | https://qualitytactics.de/en/reliability/self-monitoring-and-diagnosis/ |
| Self-Test | Ability of a component to check its own state and functionality | https://qualitytactics.de/en/reliability/self-test/ |
| Service Level Agreements | Defining expectations for software availability and performance | https://qualitytactics.de/en/reliability/service-level-agreements/ |
| Service Level Indicators | Tracking key metrics of software reliability and performance | https://qualitytactics.de/en/reliability/service-level-indicators/ |
| Service Level Objectives | Defining measurable goals for system reliability and performance | https://qualitytactics.de/en/reliability/service-level-objectives/ |
| Site Reliability Engineering (SRE) | Applying principles for stable system operations | https://qualitytactics.de/en/reliability/site-reliability-engineering-sre/ |
| Smoke Testing | Performing a series of basic tests to verify the core functionality of a system | https://qualitytactics.de/en/reliability/smoke-testing/ |
| Status Monitoring | Continuous monitoring of the condition and performance of components or services | https://qualitytactics.de/en/reliability/status-monitoring/ |
| Timeout Management | Defining and enforcing timeouts on all external calls against indefinite blocking | https://qualitytactics.de/en/reliability/timeout-management/ |
| Timestamping | Adding timestamps to data or events for temporal tracking | https://qualitytactics.de/en/reliability/timestamping/ |
| Transactions | Grouping multiple operations into an atomic, consistent unit | https://qualitytactics.de/en/reliability/transactions/ |
| Watchdog | Monitoring component for detecting and handling system errors or failures | https://qualitytactics.de/en/reliability/watchdog/ |
| Write-Ahead Logging | Recording changes in a durable append-only log before applying them | https://qualitytactics.de/en/reliability/write-ahead-logging/ |

## Security (78)

| Title | Short Description | URL |
|---|---|---|
| Abuse Case Definition | Describing undesirable use cases from the perspective of attackers | https://qualitytactics.de/en/security/abuse-case-definition/ |
| API Security | Securing APIs through rate limiting, schema validation, gateways, and token-based authentication | https://qualitytactics.de/en/security/api-security/ |
| Audit Trail Management | Maintaining tamper-proof, immutable, cryptographically chained audit records for legal and compliance purposes | https://qualitytactics.de/en/security/audit-trail-management/ |
| Authentication | Verify the identity of users and systems | https://qualitytactics.de/en/security/authentication/ |
| Authorization | Control access to resources based on permissions | https://qualitytactics.de/en/security/authorization/ |
| Authorization Concept | Defining access to critical data and functions | https://qualitytactics.de/en/security/authorization-concept/ |
| Backup and Recovery | Ensure regular backup and recoverability of data | https://qualitytactics.de/en/security/backup-and-recovery/ |
| Canonicalization | Transform input data into a canonical representation | https://qualitytactics.de/en/security/canonicalization/ |
| Certificate Management | Managing X.509 certificate lifecycles including PKI, revocation, and pinning | https://qualitytactics.de/en/security/certificate-management/ |
| Configuration Checks | Document and regularly review security-relevant settings | https://qualitytactics.de/en/security/configuration-checks/ |
| Cryptographic Methods | Use proven and standardized algorithms and protocols for cryptographic functions | https://qualitytactics.de/en/security/cryptographic-methods/ |
| Data Flow Control | Control and filter data flows between components and systems | https://qualitytactics.de/en/security/data-flow-control/ |
| Datensparsamkeit | Only collect and store personal data that is necessary for the purpose | https://qualitytactics.de/en/security/datensparsamkeit/ |
| Defense Lines | Implementing security mechanisms in multiple layers and levels | https://qualitytactics.de/en/security/defense-lines/ |
| Digital Forensics | Establishing methods for investigating security incidents and crimes | https://qualitytactics.de/en/security/digital-forensics/ |
| Digital Signatures | Using cryptographic signatures for code signing, document verification, and proving authorship | https://qualitytactics.de/en/security/digital-signatures/ |
| Dynamic Code Analysis | Testing security properties by executing and observing program behavior | https://qualitytactics.de/en/security/dynamic-code-analysis/ |
| Emergency Drills | Training behavior during security incidents and testing emergency processes | https://qualitytactics.de/en/security/emergency-drills/ |
| Encryption | Encrypt data during transmission and storage | https://qualitytactics.de/en/security/encryption/ |
| Endpoint Detection and Response | Continuously monitoring endpoints for threats in real-time | https://qualitytactics.de/en/security/endpoint-detection-and-response/ |
| Federated Identity (OAuth/OIDC) | Delegating authentication to trusted external identity providers | https://qualitytactics.de/en/security/federated-identity/ |
| Fuzz-Testing | Testing with randomly generated input data to uncover unexpected behavior | https://qualitytactics.de/en/security/fuzz-testing/ |
| Honeypots | Deploying specially secured systems as bait for attackers | https://qualitytactics.de/en/security/honeypots/ |
| Incident Response Measures | Establish processes and tools for responding to security incidents | https://qualitytactics.de/en/security/incident-response-measures/ |
| Input Validation | Validate all inputs from users and external systems | https://qualitytactics.de/en/security/input-validation/ |
| Key Management | Establish procedures for the secure generation, distribution, and storage of cryptographic keys | https://qualitytactics.de/en/security/key-management/ |
| Least Privilege | Equip users and processes with only the minimal necessary rights | https://qualitytactics.de/en/security/least-privilege/ |
| Logging and Monitoring | Log and monitor security-related events | https://qualitytactics.de/en/security/logging-and-monitoring/ |
| Malware Protection | Detect and defend against malware through technical measures | https://qualitytactics.de/en/security/malware-protection/ |
| Negative Testing | Deliberately test invalid inputs and edge cases to check error handling | https://qualitytactics.de/en/security/negative-testing/ |
| Network Segmentation | Divide the network into security zones with separate trust levels | https://qualitytactics.de/en/security/network-segmentation/ |
| Output Encoding | Mask outputs to prevent injection attacks | https://qualitytactics.de/en/security/output-encoding/ |
| Patch Management | Apply security updates and patches promptly | https://qualitytactics.de/en/security/patch-management/ |
| Penetration Tests | Uncovering security vulnerabilities through simulated attacks | https://qualitytactics.de/en/security/penetration-tests/ |
| Physical Security | Access and entry protection for IT infrastructure through structural and organizational measures | https://qualitytactics.de/en/security/physical-security/ |
| Prepared Statements | Use parameterized queries to prevent SQL injection | https://qualitytactics.de/en/security/prepared-statements/ |
| Privacy by Design | Embedding privacy protection into system architecture from inception | https://qualitytactics.de/en/security/privacy-by-design/ |
| Raising User Awareness | Sensitizing and training employees and users on security topics | https://qualitytactics.de/en/security/raising-user-awareness/ |
| Red Teaming | Conduct comprehensive and realistic attacks on your own systems | https://qualitytactics.de/en/security/red-teaming/ |
| Security Regression Tests | Retest previously fixed security vulnerabilities to prevent their recurrence | https://qualitytactics.de/en/security/regression-tests/ |
| Risk Analysis | Identifying, assessing, and addressing risks | https://qualitytactics.de/en/security/risk-analysis/ |
| Role-Based Access Control | Control access to application components based on roles | https://qualitytactics.de/en/security/role-based-access-control/ |
| Secret Management | Securely managing application secrets using dedicated vaults and rotation policies | https://qualitytactics.de/en/security/secret-management/ |
| Secure by Default | Align default settings and delivery state for maximum security | https://qualitytactics.de/en/security/secure-by-default/ |
| Secure Coding Guidelines | Define mandatory rules and best practices for secure programming | https://qualitytactics.de/en/security/secure-coding-guidelines/ |
| Secure Configuration | Deliver and operate systems with secure default settings | https://qualitytactics.de/en/security/secure-configuration/ |
| Secure Programming Interfaces | Using Libraries and Frameworks with Security Features | https://qualitytactics.de/en/security/secure-programming-interfaces/ |
| Secure Protocols | Use only secure and current versions of network protocols | https://qualitytactics.de/en/security/secure-protocols/ |
| Secure Session Management | Manage sessions based on random, time-limited ids | https://qualitytactics.de/en/security/secure-session-management/ |
| Secure Software Development | Establishing security as an integral part of the development process | https://qualitytactics.de/en/security/secure-software-development/ |
| Security Architecture Analysis | Examine architecture and design for conceptual security gaps | https://qualitytactics.de/en/security/security-architecture-analysis/ |
| Security Audits | Regularly check systems and processes for security | https://qualitytactics.de/en/security/security-audits/ |
| Security by Design | Consider security already in the design of the architecture and implementation | https://qualitytactics.de/en/security/security-by-design/ |
| Security Certification | Introduce a structured framework for assessing and improving security practices | https://qualitytactics.de/en/security/security-certification/ |
| Security Community | Promote secure software design through exchange with experts and peers | https://qualitytactics.de/en/security/security-community/ |
| Security Culture | Embedding security as a shared value within the company | https://qualitytactics.de/en/security/security-culture/ |
| Security Frameworks | Utilizing structured approaches to identify and mitigate security risks | https://qualitytactics.de/en/security/security-frameworks/ |
| Security Incident Handling | Clearly regulate processes and responsibilities for dealing with security incidents | https://qualitytactics.de/en/security/security-incident-handling/ |
| Security Metrics | Define, collect, and evaluate metrics to quantify the security status | https://qualitytactics.de/en/security/security-metrics/ |
| Security Monitoring | Continuously capture and analyze security-relevant events and data | https://qualitytactics.de/en/security/security-monitoring/ |
| Security Policies for Development | Define mandatory rules for secure software development | https://qualitytactics.de/en/security/security-policies-for-development/ |
| Security Policies for Users | Define mandatory rules for the secure usage of applications | https://qualitytactics.de/en/security/security-policies-for-users/ |
| Security-Relevant Metrics | Define and collect metrics to quantify the security level | https://qualitytactics.de/en/security/security-relevant-metrics/ |
| Security Requirements Definition | Elicit and document specific requirements for information security | https://qualitytactics.de/en/security/security-requirements-definition/ |
| Security Tests | Verify security properties through specialized testing methods | https://qualitytactics.de/en/security/security-tests/ |
| Security Tests by External Parties | Engage independent security experts to test the application | https://qualitytactics.de/en/security/security-tests-by-external-parties/ |
| Security Training | Raising awareness and further educating employees on security topics | https://qualitytactics.de/en/security/security-training/ |
| Static Code Analysis | Automatically check source code for programming errors and security vulnerabilities | https://qualitytactics.de/en/security/static-code-analysis/ |
| Supply Chain Security | Securing the software supply chain through SBOMs and provenance verification | https://qualitytactics.de/en/security/supply-chain-security/ |
| System Hardening | Improve the security state of systems and components | https://qualitytactics.de/en/security/system-hardening/ |
| Third-Party Dependency Check | Regularly review dependencies on external software | https://qualitytactics.de/en/security/third-party-dependency-check/ |
| Threat Intelligence | Collecting and analyzing information about current threats and attack methods | https://qualitytactics.de/en/security/threat-intelligence/ |
| Threat Modeling | Conduct systematic analysis of threats, attackers, and countermeasures | https://qualitytactics.de/en/security/threat-modeling/ |
| Trust Boundaries | Define boundaries between systems and components with different trust levels | https://qualitytactics.de/en/security/trust-boundaries/ |
| Two-Factor Authentication | Verify identity using two independent factors | https://qualitytactics.de/en/security/two-factor-authentication/ |
| Vulnerability Scans | Regularly check systems and applications for known vulnerabilities | https://qualitytactics.de/en/security/vulnerability-scans/ |
| Web Application Firewall | Filtering HTTP traffic at application layer against web attacks | https://qualitytactics.de/en/security/web-application-firewall/ |
| Zero Trust Architecture | "Never trust, always verify" — verifying every request regardless of network location | https://qualitytactics.de/en/security/zero-trust-architecture/ |

## Usability (53)

| Title | Short Description | URL |
|---|---|---|
| A/B Testing | Comparing different versions to optimize user experience | https://qualitytactics.de/en/usability/a-b-testing/ |
| Accessibility Concept | Design of software to make it accessible and usable for people with disabilities | https://qualitytactics.de/en/usability/accessibility-concept/ |
| Adaptive Behavior | Adjustment of system behavior based on the context, preferences, or behavior of the user | https://qualitytactics.de/en/usability/adaptive-behavior/ |
| Adjustable Font Sizes | Ability for users to adjust the font size of the user interface according to their needs | https://qualitytactics.de/en/usability/adjustable-font-sizes/ |
| Assistive Technology Support | Ensuring usability of assistive technologies | https://qualitytactics.de/en/usability/assistive-technology-support/ |
| Asynchronous Operations | Execution of time-intensive operations in the background without blocking the user interface | https://qualitytactics.de/en/usability/asynchronous-operations/ |
| Auto-Save | Automatically saving user work at regular intervals against data loss | https://qualitytactics.de/en/usability/auto-save/ |
| Cognitive Load Minimization | Designing the user interface to be intuitive and easy to understand | https://qualitytactics.de/en/usability/cognitive-load-minimization/ |
| Confirmation Dialogs for Destructive Actions | Requiring explicit user confirmation before executing irreversible operations | https://qualitytactics.de/en/usability/confirmation-dialogs/ |
| Consistent Terminology | Use uniform terms throughout the software | https://qualitytactics.de/en/usability/consistent-terminology/ |
| Consistent User Interface | Uniform design and behavior of the user interface across all parts of the software | https://qualitytactics.de/en/usability/consistent-user-interface/ |
| Contextual Help | Providing help information and explanations directly in the context of the current task | https://qualitytactics.de/en/usability/contextual-help/ |
| Custom Views | Allow users to create their own views and layouts | https://qualitytactics.de/en/usability/custom-views/ |
| Customizable User Interface | Letting the user change the user interface according to their preferences and needs | https://qualitytactics.de/en/usability/customizable-user-interface/ |
| Dark Mode | Providing an alternative dark color theme for visual comfort across environments | https://qualitytactics.de/en/usability/dark-mode/ |
| Design Tokens and Theming | Encoding visual design decisions platform-agnostically for theming and cross-platform consistency | https://qualitytactics.de/en/usability/design-tokens/ |
| Drag-and-Drop Interaction | Supporting direct manipulation through drag-and-drop for reordering, organizing, file upload, and spatial arrangement | https://qualitytactics.de/en/usability/drag-and-drop/ |
| Empty States and First-Use Guidance | Designing meaningful empty states with clear guidance on what to do next | https://qualitytactics.de/en/usability/empty-states-and-first-use-guidance/ |
| Feedback | Provision of visual or acoustic confirmations for user interactions | https://qualitytactics.de/en/usability/feedback/ |
| Feedback Mechanisms | Provide opportunities for users to submit feedback, suggestions for improvement or problem reports | https://qualitytactics.de/en/usability/feedback-mechanisms/ |
| Focus Management | Managing keyboard focus when modals open and close, ensuring visible focus indicators, and implementing proper focus traps in overlays | https://qualitytactics.de/en/usability/focus-management/ |
| Form Design and Multi-Step Wizards | Structuring complex data entry through grouped fields, multi-step wizards with progress indication, and conditional field visibility | https://qualitytactics.de/en/usability/form-design/ |
| Frequently Asked Questions (FAQ) | Providing a collection of frequently asked questions and their answers on various software topics | https://qualitytactics.de/en/usability/frequently-asked-questions-faq/ |
| Gamification | Increase motivation through playful elements | https://qualitytactics.de/en/usability/gamification/ |
| High Color Contrasts | Use of sufficient color contrasts between text and background to improve readability | https://qualitytactics.de/en/usability/high-color-contrasts/ |
| Icons | Use symbols to visually support the user interface | https://qualitytactics.de/en/usability/icons/ |
| Input Constraints and Defaults | Constraining input through dropdowns, date pickers, sliders, and sensible defaults | https://qualitytactics.de/en/usability/input-constraints-and-defaults/ |
| Integrated Onboarding | Support for new users in getting started with the software through tutorials, guides, or interactive tours | https://qualitytactics.de/en/usability/integrated-onboarding/ |
| Interactive Tutorials | Provision of interactive guides that lead users step-by-step through tasks or functions | https://qualitytactics.de/en/usability/interactive-tutorials/ |
| Intuitive Navigation | Implement a logical and easy-to-understand navigation structure | https://qualitytactics.de/en/usability/intuitive-navigation/ |
| Keyboard Support | Make the software operable via the keyboard | https://qualitytactics.de/en/usability/keyboard-support/ |
| Knowledge Base | Building a searchable knowledge base with articles, guides, and troubleshooting solutions for users | https://qualitytactics.de/en/usability/knowledge-base/ |
| Localization | Adapting software to different languages, regions, and cultural conventions | https://qualitytactics.de/en/usability/localization/ |
| Micro Interactions | Provide subtle animations and feedback to communicate system state during user actions | https://qualitytactics.de/en/usability/micro-interactions/ |
| Mobile First Design | The design of applications is primarily done for mobile devices | https://qualitytactics.de/en/usability/mobile-first-design/ |
| Performance Optimization | Improving perceived responsiveness through user-facing performance techniques | https://qualitytactics.de/en/usability/performance-optimization/ |
| Personal Support | Provision of personal support by trained staff to assist users with questions or problems | https://qualitytactics.de/en/usability/personal-support/ |
| Plain Language | Use simple and clear formulations | https://qualitytactics.de/en/usability/plain-language/ |
| Progressive Disclosure | Gradual disclosure of information and functions | https://qualitytactics.de/en/usability/progressive-disclosure/ |
| Real-time Input Validation | Verification of user inputs in real-time and provision of immediate feedback for erroneous inputs | https://qualitytactics.de/en/usability/real-time-input-validation/ |
| Responsive Design | Design of the user interface that automatically adapts to different screen sizes and device types | https://qualitytactics.de/en/usability/responsive-design/ |
| Search Function | Providing a powerful search function to find content and features quickly | https://qualitytactics.de/en/usability/search-function/ |
| Single Page Application | Provide fluid navigation and seamless transitions within a single web page | https://qualitytactics.de/en/usability/single-page-application/ |
| Style Guide | Ensure consistent design and user experience | https://qualitytactics.de/en/usability/style-guide/ |
| Subtitles and Transcripts | Provision of texts for visual and auditory content | https://qualitytactics.de/en/usability/subtitles-and-transcripts/ |
| Understandable Error Messages | Provision of clear, context-related error messages in the event of problems | https://qualitytactics.de/en/usability/understandable-error-messages/ |
| Undo and Redo | Allowing users to reverse and reapply actions for error recovery and exploration | https://qualitytactics.de/en/usability/undo-and-redo/ |
| Usability Tests | Conducting tests with representative users | https://qualitytactics.de/en/usability/usability-tests/ |
| User-Centered Design | Incorporate users' needs, expectations, and abilities from the beginning | https://qualitytactics.de/en/usability/user-centered-design/ |
| User Communities | Establish a platform for exchange and support among users | https://qualitytactics.de/en/usability/user-communities/ |
| Video Tutorials | Provision of video tutorials that visually demonstrate features and workflows | https://qualitytactics.de/en/usability/video-tutorials/ |
| Visual Hierarchy | Highlight important elements on the user interface and create a clear visual structure | https://qualitytactics.de/en/usability/visual-hierarchy/ |
| Wireframing | Create preliminary visual representations as a basis for discussion | https://qualitytactics.de/en/usability/wireframing/ |

*Total: 539 tactics*