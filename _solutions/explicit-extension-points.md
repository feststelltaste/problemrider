---
title: Explicit Extension Points
description: Define a bounded, versioned set of places where customer-specific behavior may attach, so that variation lives at the edges instead of throughout the core.
category:
- Architecture
- Business
- Code
problems:
- excessive-customization
- entity-attribute-value-overuse
- high-technical-debt
- testing-complexity
- slow-feature-development
- increased-cost-of-development
- long-release-cycles
- regression-bugs
- tight-coupling-issues
- high-maintenance-costs
- knowledge-silos
- feature-creep
- eager-to-please-stakeholders
- schema-evolution-paralysis
- core-modification-of-standard-software
- upgrade-blocked-by-customization
- voided-vendor-support
layout: solution
---

## Description

Explicit extension points are a defined, bounded, versioned set of places where customer-specific or site-specific behavior is permitted to attach — a set of hooks, a rules interface, a defined configuration surface — with the rule that variation may exist only there. The alternative, which is what most heavily customized systems actually have, is unlimited customization: any part of the core may be conditionally altered for any installation, so variation is distributed everywhere and the core has no definition. The distinction determines whether a product remains upgradeable. When variation lives behind a stated interface, the core can be changed freely as long as the interface holds, and every installation can take the change. When it is woven through the core, every change must be reasoned about against every variant, which is the condition that eventually stops the product evolving at all.

## How to Apply ◆

> The decisive question is not whether to allow customization but where it is allowed to live, and most legacy products have never answered it.

- **Define the core first.** What the standard product does, without qualification. Until this exists as a stated thing, no request can be evaluated as inside or outside it, and every request is therefore inside it.
- **Enumerate the extension points deliberately** and keep the set small. Each one is a commitment you will maintain across versions, so a large surface is a large permanent liability. Derive them from what customers have actually needed, using the existing customizations as evidence.
- **Version and document them like a public interface**, because that is what they are. A customer's extension breaking on upgrade is a support incident whether or not you consider the interface internal.
- **Make the core unable to know about specific customers.** No conditional keyed to an installation identifier, anywhere. This rule is the operational form of the whole practice, and it is checkable — a search for such conditionals measures compliance directly.
- **Route new requests to the extension mechanism**, and where a request cannot be met that way, treat that as a design question about whether the extension surface is missing something rather than as licence to modify the core.
- **Give extensions their own tests**, owned alongside them, so a customer's variation is verified independently rather than expanding the core's test matrix.
- **Migrate the existing customizations gradually.** Take the ones in the areas being changed anyway, and move them behind the extension surface as part of that work. A wholesale migration will not be funded; an opportunistic one accumulates.
- **Publish what the extension points do not cover.** Being explicit about what cannot be customized is as valuable as the interface itself, because it is what allows a request to be declined with a reason rather than by negotiation.
- **Review the surface periodically.** Extension points that nothing uses should be removed; those that every customer uses identically are candidates for promotion into the core, since a universally used extension is product functionality wearing a disguise.

## Tradeoffs ⇄

> A bounded extension surface is what keeps a customizable product upgradeable, but designing it well is genuinely hard and the boundary will be tested constantly.

**Benefits:**

- The core can evolve freely, because changes only need to respect the extension contract rather than every installation's variant.
- Upgrades become routine again, which reverses the pattern where customers fall behind and their installations drift further apart.
- The test matrix stops multiplying, since extensions are tested against the interface rather than every combination being verified in the core.
- Requests acquire a stated answer — inside the surface, outside it, or a reason to extend the surface — instead of being settled by whoever pushes hardest.
- Customer-specific code becomes locatable and attributable, rather than distributed through the core where nobody can find or cost it.

**Costs and Risks:**

- Designing extension points requires anticipating what will need to vary, and points designed wrongly are worse than none — they constrain without enabling.
- The surface is a permanent commitment. Once customers build against it, changing it is a breaking change with all the coordination that implies.
- Some genuine requests will fall outside any reasonable surface, and the discipline requires occasionally saying no to revenue.
- Migrating existing customizations is slow, and during the transition the system carries both patterns.
- Over-general extension mechanisms drift toward becoming a programming environment inside the product, at which point the customizations are as opaque as the core modifications they replaced.

## How It Could Be

A vendor of warehouse software had customer-specific logic in 60-odd places across their core, keyed to a site identifier. Upgrading a site took two to four weeks of consultancy, and eleven of 34 sites were more than a year behind. They defined the core explicitly for the first time — a two-week exercise that was mostly argument — and derived seven extension points from what the existing customizations actually did: three rule hooks in the picking flow, a document template mechanism, two event handlers, and a defined configuration schema. New requests went to that surface. Existing customizations were migrated opportunistically whenever their area was touched. After eighteen months, 44 of the 60 had moved, no site conditional remained in the picking flow, and the upgrade for a migrated site had fallen to under a day.

The rule that the core may not name a customer turned out to be the enforceable part. It was checkable in the build — a search for site identifiers in core modules — and it failed the build four times in the first two months, each time catching a developer taking the familiar shortcut under deadline pressure. The team's assessment was that the extension points themselves were only half the intervention; the other half was that "add a conditional for this customer" had stopped being an available option, which forced the design conversation that the extension surface then answered.
