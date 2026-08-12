---
title: Application Portfolio Inventory
description: Maintain a single record of what systems exist, what they do, who owns them, what they depend on, and what condition they are in.
category:
- Architecture
- Management
- Dependencies
problems:
- system-integration-blindness
- hidden-dependencies
- technology-stack-fragmentation
- obsolete-technologies
- unclear-documentation-ownership
- lack-of-ownership-and-accountability
- vendor-dependency
- knowledge-gaps
- monitoring-gaps
- legacy-system-documentation-archaeology
- shared-dependencies
- high-maintenance-costs
- accumulated-decision-debt
- communication-risk-outside-project
- dependency-on-supplier
- information-decay
- legacy-configuration-management-chaos
- legal-disputes
- modernization-roi-justification-failure
- technical-architecture-limitations
- vendor-relationship-strain
layout: solution
---

## Description

An application portfolio inventory is a single maintained record of the systems an organization runs: what each one does, who owns it, what technology it uses, what it depends on and what depends on it, its business criticality, and its condition. It sounds like administration and it is the precondition for almost every strategic decision about a legacy landscape. Organizations of any age routinely cannot answer basic questions — how many systems do we have, which ones handle personal data, what would break if this database were unavailable — and every modernization plan, risk assessment, and impact analysis begins by partially reconstructing the answer for that occasion. The inventory replaces a dozen partial reconstructions with one maintained record. Its value is proportional to how disorganized the landscape is, which means it is most valuable exactly where it is most difficult to build.

## How to Apply ◆

> The systems missing from any organization's mental map are the ones nobody has thought about in years, which is a strong predictor of which ones are about to cause a problem.

- **Discover rather than survey.** Start from infrastructure inventories, network traffic, DNS records, certificate registries, authentication logs, licence records, and firewall rules. Asking teams what they run finds what they remember; automated discovery finds what they have forgotten.
- **Keep the record deliberately small.** Ten to fifteen fields, no more: name, purpose in one sentence, owning team, technology, criticality, upstream and downstream dependencies, data classification, support status, and last reviewed date. Ambitious schemas produce inventories that are ninety percent empty.
- **Record dependencies in both directions.** What this system calls and what calls it. The inbound direction is harder to establish and is the one needed for impact analysis and decommissioning, which is why it is usually the one missing.
- **Require a named owning team per system**, and treat any system without one as a finding rather than a gap in the record. Unowned systems are where incidents and unpatched vulnerabilities concentrate.
- **Attach the operational reality** — support status, end-of-support dates, last patched, whether monitoring exists, whether a recovery procedure has been tested. This is what turns an inventory from a catalogue into a risk register.
- **Tie it to a process that keeps it current**, or accept that it will be stale within a year. Onboarding a new system, decommissioning one, and any change of ownership must update the record, and the update has to be part of the process rather than an act of virtue.
- **Review a rotating slice periodically** rather than attempting a full refresh. Twenty entries a quarter keeps a 200-system inventory roughly current with modest effort; an annual full review is scheduled, deferred, and never done.
- **Make it the accepted single source.** An inventory competing with three spreadsheets in different departments is a fourth spreadsheet. Consolidating the existing partial records is usually the first and most political piece of work.
- **Publish it broadly.** Its value comes from being consulted, and it is consulted only if people know it exists and can search it without asking permission.
- **Use it to drive decisions**, not merely to describe: candidates for decommissioning, unowned systems, unsupported technology, and single points of failure all fall directly out of the fields above.

## Tradeoffs ⇄

> An inventory is the foundation for every strategic decision about a legacy landscape, but building it is unglamorous and keeping it current requires permanent discipline.

**Benefits:**

- Impact analysis, decommissioning, and modernization planning all start from a known baseline instead of a partial reconstruction repeated each time.
- Unowned systems and unsupported technologies become visible, and these are consistently where incidents and security exposures concentrate.
- Bidirectional dependency records make the blast radius of a change or an outage assessable in advance.
- Compliance and audit questions — where personal data lives, what is in scope for a given regulation — become answerable in hours rather than weeks.
- Retirement candidates surface automatically from the combination of low criticality, unsupported technology, and no owner.

**Costs and Risks:**

- The initial discovery is substantial work with no immediate output, and it is hard to fund against anything that delivers a feature.
- Inventories go stale rapidly, and a stale one is dangerous because decisions get made on it — an entry naming an owner who left two years ago is worse than a blank field.
- Over-ambitious schemas collapse under their own weight, producing a mostly empty record that nobody trusts or maintains.
- Ownership fields make accountability explicit, which is resisted in organizations where being named as owner means inheriting a problem with no resources.
- The inventory can become an end in itself, consuming effort in maintenance and reporting that exceeds the decisions it informs.

## How It Could Be

An organization believed it ran approximately 60 applications. Discovery from certificate records, network flow logs, and licence data found 143 distinct running systems, including a customer-facing web application nobody in the current organization was aware of — a legacy self-service portal from an acquisition six years earlier, still reachable from the internet, still authenticating against a directory, and last patched in 2019. It had no owner, no monitoring, and no entry in any list. Its discovery was the single strongest argument for funding the inventory work, which was completed for the remaining 142 systems over two quarters. Nineteen systems turned out to have no identifiable owner, and eleven were running technology past vendor support.

The bidirectional dependency records changed the organization's incident response. A database cluster required emergency maintenance, and the previous approach would have been to notify everyone and hope. The inventory listed nine systems with a recorded dependency on that cluster, of which three were classified business-critical. The maintenance was scheduled around those three specifically, with their owning teams involved in advance. During the window, a tenth system failed — one whose dependency was undocumented, added eight months earlier by a team that had not updated the record. That failure became the reason dependency updates were made a mandatory step in the change process rather than a courtesy, which is the kind of enforcement an inventory needs and rarely gets before something demonstrates why.
