---
title: Voided Vendor Support
description: The vendor declines to diagnose problems because the installation is modified, so the organization pays for support it can no longer use.
category:
- Dependencies
- Operations
- Business
related_problems:
solutions:
- vendor-management-practice
- explicit-extension-points
- fit-to-standard-principle
- service-level-agreements
- risk-quantification
- total-cost-of-ownership-transparency
- application-portfolio-inventory
- knowledge-rotation
- written-first-communication
layout: problem
---

## Description

Voided vendor support occurs when local modification of a packaged product removes the organization's practical access to the support it is paying for. The refusal is rarely absolute. More often it is procedural: the vendor asks for the problem to be reproduced on an unmodified installation, which the organization cannot provide, or narrows its responsibility to the delivered code, which is not what is running. The effect is the same. The organization continues paying a support fee, and every incident is diagnosed internally by people with far less knowledge of the product than the vendor has. Because the refusal happens per incident rather than as a formal withdrawal, the position is often not recognized as a decision anyone made — it accumulated.

## Indicators ⟡

- Support tickets are routinely closed with a request to reproduce on a standard system
- The team no longer opens vendor tickets for certain modules because the outcome is predictable
- Incidents in modified areas are diagnosed entirely internally, regardless of severity
- Nobody can state what the support contract actually covers given the current installation state
- The renewal is approved annually without anyone assessing what value was obtained
- Escalations require the account manager rather than the support process

## Symptoms ▲

- [Slow Incident Resolution](slow-incident-resolution.md)
<br/>  Problems that the vendor could diagnose from experience are worked out locally from first principles, which takes far longer.
- [Legacy Skill Shortage](legacy-skill-shortage.md)
<br/>  The organization must sustain deep product knowledge internally, in a product it did not write, because it can no longer draw on the vendor's.
- [High Maintenance Costs](high-maintenance-costs.md)
<br/>  A support fee is paid for a service that cannot be used, while the diagnostic work it was meant to cover is done at internal cost.
- [Constant Firefighting](constant-firefighting.md)
<br/>  Incidents in modified areas consume disproportionate effort, and their frequency does not fall because the underlying causes are never properly diagnosed.
- [Vendor Relationship Strain](vendor-relationship-strain.md)
<br/>  Repeated refusals produce an adversarial dynamic in which each side regards the other as acting in bad faith.
- [Vendor Dependency Entrapment](vendor-dependency-entrapment.md)
<br/>  The organization depends on a product whose maker will not help it, and cannot leave because the modifications would have to be rebuilt.

## Causes ▼

- [Core Modification of Standard Software](core-modification-of-standard-software.md)
<br/>  Modification of delivered code is the specific condition most support agreements exclude, and it is usually what triggers the refusal.
- [Upgrade Blocked by Customization](upgrade-blocked-by-customization.md)
<br/>  Running an out-of-support version removes the entitlement entirely, independently of any modification.
- [Poor Contract Design](poor-contract-design.md)
<br/>  What the vendor will and will not support given a modified installation is frequently unaddressed at signing and discovered during an incident.
- [Excessive Customization](excessive-customization.md)
<br/>  The volume of adaptation makes it impractical to reproduce any problem on a standard installation, which is what the vendor requires.
- [Legal Disputes](legal-disputes.md)
<br/>  Where the relationship has become contentious, the support boundary tends to be interpreted narrowly by both parties.

## Detection Methods ○

- Review the last year of vendor tickets and count how many were closed without a diagnosis on modification or version grounds
- Ask the support team which modules they no longer raise tickets for, and why
- Read the support agreement against the current installation state and identify what is actually covered
- Compare the support fee against the value obtained, measured as tickets successfully resolved by the vendor
- Measure incident resolution time in modified versus unmodified areas of the product
- Ask whether the organization could reproduce a given problem on a clean installation, and how long that would take

## Examples

An organization running a heavily adapted enterprise resource planning system paid an annual support fee in the high six figures. A review of the preceding year's tickets found that of 94 raised, 61 had been closed with a request to reproduce on an unmodified system. The team had stopped raising tickets for three modules entirely. Nobody had ever computed the effective cost per resolved ticket, and when it was computed it was roughly forty times what the organization assumed it was paying. The renewal had been approved for eleven consecutive years on the basis that support for a critical system was obviously necessary — which it was, and which was not what they were receiving.

A document management deployment showed how the position accumulates rather than being chosen. A single modification made in 2016 to a retention routine placed that module outside support. This was noted at the time in a ticket comment and never escalated. Over the following years the affected area grew as further changes were made around the original one. By the time an incident in that area caused a genuine compliance exposure, the boundary of what the vendor would help with had moved substantially, and no document existed recording where it now lay or who had agreed to it.
