---
title: Technologie-Lock-in
description: Eine Situation, in der es schwierig oder unmöglich ist, zu einer neuen
  Technologie zu wechseln, aufgrund der hohen Kosten oder des erforderlichen Aufwands.
category:
- Architecture
- Code
related_problems:
- slug: vendor-lock-in
  similarity: 0.7
- slug: technology-isolation
  similarity: 0.6
- slug: system-stagnation
  similarity: 0.6
- slug: integration-difficulties
  similarity: 0.55
- slug: difficult-code-reuse
  similarity: 0.55
- slug: vendor-dependency-entrapment
  similarity: 0.55
solutions:
- anti-corruption-layer
- dependency-management-strategy
- strangler-fig-pattern
- abstracted-file-system-access
- abstraction
- abstraction-layers
- bridges
- browser-compatibility
- cloud-native-development
- cross-platform-build-scripts
- cross-platform-build-tools
- cross-platform-frameworks
- data-export
- emulation
- feature-detection
- forward-compatibility
- hexagonal-architecture
- microservices-architecture
- multi-cloud-iac
- object-relational-mapping-orm
- platform-independence
- platform-independent-build-pipelines
- platform-independent-data-storage
- platform-independent-programming-languages
- platform-independent-scripting-languages
- portability-checklists
- protocol-abstraction
- secure-programming-interfaces
- secure-protocols
- standard-software
- standardized-data-formats
- standardized-interfaces
- standardized-protocols
- virtual-networks
- virtualization
- database-abstraction
- dependency-injection
- federated-identity
- patch-management
- supply-chain-security
- third-party-dependency-check
- vendor-management-practice
- technology-radar
- system-decommissioning
- modernization-options-comparison
- no-regret-moves
- risk-quantification
- cost-of-delay
- staged-investment-with-decision-gates
- automated-code-migration
- continuous-dependency-updates
layout: problem
lang: de
en_slug: technology-lock-in
---

## Description
Technologie-Lock-in ist eine Situation, in der es schwierig oder unmöglich ist, zu einer neuen Technologie zu wechseln, aufgrund der hohen Kosten oder des erforderlichen Aufwands. Dies ist ein häufiges Problem in monolithischen Architekturen, wo das gesamte System auf einem einzigen Technologie-Stack aufgebaut ist. Technologie-Lock-in kann Innovation erschweren und auch zu hohen Kosten führen, wenn die Technologie veraltet oder der Anbieter aufgibt.

## Indicators ⟡
- Das gesamte System ist auf einem einzigen Technologie-Stack aufgebaut.
- Es ist schwierig oder unmöglich, neue Technologien im System zu nutzen.
- Das Entwicklungsteam kann mit den neuesten Technologietrends nicht Schritt halten.
- Das System ist teuer in der Wartung aufgrund der hohen Kosten der Technologie.

## Symptoms ▲

- [Technologie-Isolation](technologie-isolation.md)
<br/>  In einen bestimmten Technologie-Stack eingesperrt zu sein, verhindert die Übernahme moderner Alternativen und isoliert das System von aktuellen Ökosystemen.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Proprietäre oder veraltete eingesperrte Technologien haben oft hohe Lizenz- und Supportkosten.
- [Verringerte Innovation](verringerte-innovation.md)
<br/>  Die Unfähigkeit, neue Technologien zu übernehmen, schränkt die Fähigkeit des Teams ein, zu innovieren und moderne Fähigkeiten zu nutzen.
- [Systemstagnation](systemstagnation.md)
<br/>  Die Unfähigkeit, den Technologie-Stack weiterzuentwickeln, verursacht, dass das System stagniert und hinter Wettbewerber zurückfällt.

## Causes ▼

- [Einschränkungen durch monolithische Architektur](einschraenkungen-durch-monolithische-architektur.md)
<br/>  Eine monolithische Architektur, die auf einem einzigen Technologie-Stack aufgebaut ist, macht es unmöglich, neue Technologien inkrementell zu übernehmen.
- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Wenn Code eng an spezifische Technologie-APIs und -Muster gekoppelt ist, erfordert der Technologiewechsel das Umschreiben großer Teile des Systems.
- [Angehäufte Entscheidungsschulden](angehaeufte-entscheidungsschulden.md)
<br/>  Frühe Technologieentscheidungen, die nie überarbeitet wurden, werden tief eingebettet, was Änderungen zunehmend teuer macht.

## Detection Methods ○
- **Technologie-Stack-Analyse:** Analyse des Technologie-Stacks des Systems zur Identifikation, welche Technologien genutzt werden.
- **Entwicklerbefragungen:** Befragung von Entwicklern, ob sie das Gefühl haben, neue Technologien zur Verbesserung des Systems nutzen zu können.
- **Kostenanalyse:** Analyse der Kosten der Technologie zur Identifikation, welche Technologien am teuersten sind.

## Examples
Ein Unternehmen hat eine große, monolithische E-Commerce-Anwendung, die auf einem proprietären Technologie-Stack aufgebaut ist. Das Unternehmen kann keine neuen Technologien wie Cloud Computing und Microservices nutzen, weil das System nicht dafür designt ist. Infolgedessen kann das Unternehmen nicht so schnell innovieren wie seine Wettbewerber. Das Unternehmen zahlt außerdem viel Geld für die proprietäre Technologie und ist an einen einzigen Anbieter gebunden.
