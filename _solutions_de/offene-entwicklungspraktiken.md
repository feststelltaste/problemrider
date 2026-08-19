---
title: Offene Entwicklungspraktiken
description: Verbesserung der Codequalität durch öffentliches Code-Review, transparentes
  Issue-Tracking und externe Beiträge.
category:
- Process
- Culture
problems:
- knowledge-silos
- insufficient-code-review
- poor-documentation
- limited-team-learning
- resistance-to-change
- feedback-isolation
- team-silos
layout: solution
lang: de
en_slug: fair-source
related_solutions:
- slug: code-review-process-reform
  similarity: 0.8
- slug: code-reviews
  similarity: 0.75
- slug: pair-and-mob-programming
  similarity: 0.7
- slug: code-metrics
  similarity: 0.7
- slug: psychological-safety-practices
  similarity: 0.7
- slug: static-analysis-and-linting
  similarity: 0.7
---

## Description

Diese Lösung öffnet den Entwicklungsprozess einer Codebasis — öffentliches Code-Review, transparentes Issue-Tracking und strukturierte Beitragsrichtlinien — für ein breiteres Publikum als das Team, dem sie historisch exklusiv gehörte, sei es der Rest der Organisation oder, bei Open-Source-Projekten, die Öffentlichkeit insgesamt. Legacy-Systeme neigen besonders dazu, sowohl Wissen als auch Review-Kapazität in einem einzigen kleinen Team zu konzentrieren, sodass technische Schulden und bekannte Defekte jahrelang still in einem privaten Backlog liegen, unsichtbar für jeden, der sonst die freie Kapazität oder frische Perspektive hätte, sie anzugehen. Die Codebasis, ihre Issues und ihre Pull Requests breit sichtbar zu machen — mit klaren Beitragsrichtlinien und markierten Einstiegspunkten im Stil von „Good First Issue" — lädt externe Reviewer und gelegentliche Beitragende ein, Probleme zu finden, die dem besitzenden Team zu vertraut geworden sind, um sie noch zu sehen, und Aufräumarbeiten zu erledigen, die es in einem internen Backlog nie an die Spitze schaffen. Die offensichtlichen Kosten sind Governance: Eingehende Beiträge brauchen Review-Aufwand und Qualitätsschranken, um eine Verschlechterung der Codebasis zu vermeiden, mancher Legacy-Code mag von vornherein zu sicherheitskritisch für breite Sichtbarkeit sein, und die öffentliche Offenlegung des tatsächlichen Zustands einer Codebasis kann für das Team, das sie gepflegt hat, selbst eine unangenehme Anpassung sein.

## How to Apply ◆

> In Legacy-System-Kontexten erhöhen offene Entwicklungspraktiken die Transparenz und ziehen frische Perspektiven an, die eingefahrene Annahmen darüber, wie das System funktionieren muss, infrage stellen können.

- Machen Sie die Codebasis einem breiteren Publikum innerhalb der Organisation (oder extern bei Open-Source-Projekten) zugänglich, indem Sie sie auf Plattformen hosten, die Code-Review, Issue-Tracking und Beiträge unterstützen.
- Etablieren Sie Beitragsrichtlinien, die klarstellen, wie externe Beitragende (aus anderen Teams oder außerhalb der Organisation) Probleme melden, Verbesserungen vorschlagen und Änderungen einreichen können.
- Nutzen Sie öffentliches Issue-Tracking, um technische Schulden, bekannte Fehler und Verbesserungsmöglichkeiten sichtbar zu machen statt in privaten Backlogs zu verstecken.
- Fördern Sie teamübergreifendes Code-Review, indem Pull Requests für jeden in der Organisation sichtbar und überprüfbar sind, nicht nur für das besitzende Team.
- Dokumentieren Sie architektonische Entscheidungen, Coding-Konventionen und Systembeschränkungen öffentlich, damit potenzielle Beitragende sich selbst einarbeiten können.
- Erstellen Sie „Good First Issue"-Labels für Legacy-System-Aufräumaufgaben, die externe Beitragende ohne tiefes Systemwissen angehen können.

## Tradeoffs ⇄

> Offene Entwicklungspraktiken erhöhen die Transparenz und ziehen Beiträge an, erfordern aber Governance und Qualitätskontrolle für eingehende Änderungen.

**Vorteile:**

- Bringt frische Perspektiven in Legacy-Code, der von externen Blickwinkeln profitieren kann, die nicht durch Jahre angesammelter Annahmen belastet sind.
- Erhöht die Code-Review-Abdeckung, indem Code für einen breiteren Pool an Reviewern sichtbar wird.
- Verbessert die Dokumentationsqualität, weil öffentlicher Code für Menschen ohne institutionellen Kontext verständlich sein muss.
- Verringert Wissenssilos, indem Code, Entscheidungen und Diskussionen für alle transparent werden.

**Kosten und Risiken:**

- Öffentliche Sichtbarkeit der Legacy-Code-Qualität kann bei den für den Code verantwortlichen Teams Verlegenheit oder Widerstand auslösen.
- Externe Beiträge erfordern Review-Aufwand und erfüllen ohne klare Beitragsrichtlinien möglicherweise keine Qualitätsstandards.
- Sicherheitskritischer Legacy-Code ist möglicherweise nicht für breite Sichtbarkeit geeignet.
- Das Pflegen offener Entwicklungsinfrastruktur und das Reagieren auf Community-Beiträge erfordert dediziertes Engagement.

## How It Could Be

> Das folgende Szenario veranschaulicht, wie offene Entwicklungspraktiken ein Legacy-System verbessern.

Ein Großunternehmen mit 20 Entwicklungsteams pflegte ein gemeinsam genutztes Legacy-Framework, von dem alle Teams abhingen, das aber offiziell nur ein Team besaß. Durch den Umzug des Frameworks in ein internes offenes Entwicklungsmodell mit öffentlichen Pull Requests und Issue-Tracking ermöglichte das Unternehmen anderen Teams, Fixes und Verbesserungen direkt beizutragen, statt im Backlog des besitzenden Teams zu warten. Im ersten Jahr trugen 14 Teams 120 Pull Requests bei — 80 % davon waren Bugfixes und Dokumentationsverbesserungen, die das besitzende Team nie priorisiert hatte. Der transparente Issue-Tracker enthüllte zudem, dass drei Teams unabhängig voneinander Workarounds für dieselbe Framework-Einschränkung gebaut hatten, was zu einem koordinierten Fix führte, von dem alle profitierten.
