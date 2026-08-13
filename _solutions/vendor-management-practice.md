---
title: Vendor Management Practice
description: Treat every external supplier as a managed risk with a named owner, tracked
  obligations, tested exit options, and a relationship that survives disagreement.
category:
- Dependencies
- Management
- Business
problems:
- dependency-on-supplier
- vendor-relationship-strain
- legal-disputes
- poor-contract-design
- technology-isolation
- vendor-lock-in
- technology-lock-in
- obsolete-technologies
- integration-difficulties
- vendor-dependency-entrapment
- core-modification-of-standard-software
- implementation-partner-dependency
- voided-vendor-support
layout: solution
related_solutions:
- slug: application-portfolio-inventory
  similarity: 0.65
- slug: dependency-management-strategy
  similarity: 0.6
- slug: continuous-dependency-updates
  similarity: 0.6
- slug: knowledge-sharing-practices
  similarity: 0.6
- slug: system-decommissioning
  similarity: 0.6
- slug: technology-radar
  similarity: 0.55
---

## Description

A vendor management practice is the set of routines by which an organization keeps external suppliers — software vendors, hosting providers, outsourced development partners, and licensed components — from becoming unmanaged risks. It has four parts: someone is accountable for each supplier relationship, the obligations on both sides are recorded and actually checked, the cost and feasibility of leaving is known rather than assumed, and the relationship is maintained deliberately rather than only during escalations. Legacy systems accumulate supplier dependencies over decades, and they accumulate them badly: the person who negotiated the contract has left, the contract itself is difficult to locate, nobody has assessed what replacing the component would cost, and the relationship consists entirely of an annual invoice and occasional angry emails. Each of those is individually manageable and collectively is how organizations end up unable to move.

## How to Apply ◆

> In a legacy landscape the critical suppliers are often the ones nobody thinks about — a licensed library embedded in 2009, a data feed with no contract anyone can find — so the practice starts with an inventory rather than with negotiation.

- **Inventory every external dependency** that would cause a problem if it stopped: commercial software, libraries with restrictive licences, data feeds, hosted services, and outsourced development. For each, record what it does, what it costs, who owns the relationship internally, where the contract is, and when it renews. Many organizations cannot answer these questions for a majority of their suppliers, and finding that out is itself the first result.
- **Assign a named internal owner** per supplier — not a department. The owner is responsible for knowing the contractual terms, tracking whether obligations are met, and being the point of contact. Unowned supplier relationships are the ones that surface as crises.
- **Classify by criticality and substitutability.** A supplier that is both critical and hard to replace warrants active management: known exit path, tested alternatives, escalation contacts. One that is easily substitutable needs almost nothing. Applying uniform rigor to all suppliers guarantees that the important ones get the same insufficient attention as the trivial ones.
- **Know the exit cost before you need it.** For each critical supplier, document what replacement would involve, roughly what it would cost, and how long it would take. This need not be a migration plan; an honest one-page estimate transforms negotiating position, because a party that does not know its alternatives has none.
- Negotiate for **data and interface portability** rather than for price alone: export in a documented format, source code escrow where appropriate, a defined notice period, and the right to run acceptance tests. These terms cost little at signing and are unobtainable later.
- **Verify obligations rather than assuming them.** Service levels, support response times, and security commitments should be checked against actual performance on a schedule. Unverified contractual promises are frequently unmet, and the discovery usually happens during an incident.
- **Track end-of-support dates centrally** and treat them as planning inputs a year ahead. Running unsupported components is a decision, and it should be made explicitly with the risk stated rather than arrived at by inattention.
- **Maintain the relationship outside of escalations.** A scheduled quarterly conversation with a critical supplier, when nothing is wrong, produces better outcomes when something is. Relationships that consist only of complaints degrade into positional conflict, and disputes are far more expensive than the meetings that would have prevented them.
- **Escalate contractually before escalating legally.** Document issues in writing as they occur, invoke the contract's own remedy process, and keep a record. Most supplier disputes that reach lawyers do so because nothing was documented while it was still fixable.

## Tradeoffs ⇄

> Managing suppliers properly costs ongoing administrative effort and some goodwill, in exchange for not discovering during a crisis that you have no options.

**Benefits:**

- Supplier failures, price increases, and discontinuations stop being emergencies, because the alternatives and their costs are already known.
- Negotiating position improves substantially, since a documented exit path is the only real leverage a customer has.
- Lock-in is identified while it is still cheap to address, rather than at the point where a supplier's decision has become the organization's constraint.
- Disputes are less frequent and less severe, because obligations are tracked and problems are documented while they are still routine.
- End-of-support surprises largely disappear, which removes one of the more common causes of unplanned, urgent modernization work.

**Costs and Risks:**

- The inventory and the ongoing tracking are real administrative work with no visible output, and they are easy to deprioritize until an incident makes them retrospectively obvious.
- Exit assessments cost effort for options that are usually never exercised, and it is difficult to justify keeping them current.
- Portability requirements and escrow terms raise the price of a contract, sometimes significantly, and the benefit is contingent.
- Formalizing a relationship can be read as distrust, particularly with small suppliers or long-standing partners where the informal relationship has worked well.
- Maintaining a tested alternative to a critical supplier is genuinely expensive, and for many dependencies the honest answer is to accept the risk and document that acceptance rather than to mitigate it.

## How It Could Be

A financial services firm relied on a third-party rules engine embedded in its core system since 2011. The vendor announced end of support with twelve months' notice. Nobody internally owned the relationship, the contract took two weeks to locate, and it emerged that no export format for the accumulated rule definitions was contractually guaranteed. Extracting roughly 4,000 rules from a proprietary binary format consumed nine months and two full-time developers. In the aftermath the firm inventoried its 60-odd external dependencies, identified nine as critical and hard to replace, and produced a one-page exit assessment for each. The first assessment they wrote — for a document generation service — revealed that migrating away would take three months, which they used the following year to negotiate a renewal at 30 percent below the vendor's opening position.

A different organization avoided a dispute through documentation rather than escalation. Their outsourced maintenance partner had been missing the contractual four-hour response target for critical incidents, but the misses were noticed individually and forgotten. After assigning an owner who tracked response times against the agreement, the pattern became visible: 40 percent of critical incidents over six months had exceeded the target. Presented in a scheduled quarterly review as data rather than as an accusation, it led the supplier to restructure their on-call coverage for the account. The alternative path, which the organization's legal team had begun preparing, would have been a contractual dispute over an aggregate of individually deniable incidents.
