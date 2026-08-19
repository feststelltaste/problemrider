---
title: Architektur-Workshops
description: Durchführung regelmäßiger Workshops zur Weiterentwicklung der Softwarearchitektur.
category:
- Architecture
- Team
problems:
- stagnant-architecture
- knowledge-silos
- implicit-knowledge
- team-silos
- limited-team-learning
- architectural-mismatch
- modernization-strategy-paralysis
- resistance-to-change
layout: solution
lang: de
en_slug: architecture-workshops
related_solutions:
- slug: architecture-reviews
  similarity: 0.75
- slug: architecture-documentation
  similarity: 0.75
- slug: architecture-roadmap
  similarity: 0.75
- slug: architecture-decision-records
  similarity: 0.75
- slug: architecture-conformity-analysis
  similarity: 0.7
- slug: architecture-review-board
  similarity: 0.7
---

## Description

Architektur-Workshops sind wiederkehrende, strukturierte Sitzungen — typischerweise monatlich oder vierteljährlich —, in denen Entwickler aus verschiedenen Teams gemeinsam die Architektur eines gemeinsamen Systems durch praktische Aktivitäten wie kollaboratives Diagrammieren oder geführte Codebasis-Erkundung untersuchen, dokumentieren und Änderungen vorschlagen, statt passiver Statuspräsentationen. In Legacy-Systemen, die von mehreren Teams gepflegt werden, tendiert das Verständnis der Architektur dazu, entlang von Teamgrenzen zu fragmentieren: Jede Gruppe versteht die Teile des Systems, die sie täglich berührt, tiefgehend, hat aber nur ein teilweises, manchmal veraltetes Bild davon, wie diese Teile mit allem anderen zusammenhängen, und niemand hält das vollständige Bild allein. Menschen aus verschiedenen Teams in denselben Raum zu bringen, um gemeinsam Datenflüsse zu kartieren oder ein bestimmtes architektonisches Anliegen zu diskutieren, bringt genau diese Art fragmentierten Wissens ans Licht — versteckte zirkuläre Abhängigkeiten, undokumentierte Integrationspunkte und auseinanderdriftende mentale Modelle davon, wie das System tatsächlich funktioniert —, was keine internen Meetings eines einzelnen Teams von sich aus offenbaren würden. Weil jeder Workshop sich auf ein konkretes architektonisches Anliegen fokussiert statt auf eine ergebnisoffene Diskussion, produziert er umsetzbare Ergebnisse wie einen konkreten Plan, einen entdeckten Abhängigkeitszyklus zu durchbrechen, statt eines allgemeinen Gesprächs, das ins Leere läuft. Dies macht die Workshops zu einem kostengünstigen, wiederkehrenden Mechanismus zum Aufbau gemeinsamen architektonischen Verständnisses und zur Erzeugung teamübergreifender Dynamik hin zu einer gemeinsam verstandenen Zielarchitektur, was oft eine Voraussetzung für den Erfolg jeder koordinierten Modernisierungsbemühung ist. Die Hauptkosten sind das gleichzeitige Zeitengagement mehrerer Teammitglieder und der Bedarf an geschickter Moderation, da Workshops ohne beides Gefahr laufen, zu unfokussierter Debatte zu entarten, die keine Folgemaßnahmen produziert.

## How to Apply ◆

> In Legacy-Umgebungen brechen Architektur-Workshops Wissenssilos auf und bauen gemeinsames Verständnis sowohl des aktuellen Systems als auch der Zielarchitektur auf.

- Planen Sie regelmäßige Workshops (monatlich oder vierteljährlich), in denen Entwickler aus verschiedenen Teams die Architektur des Systems gemeinsam untersuchen, diskutieren und Verbesserungen vorschlagen.
- Nutzen Sie Workshops, um schlecht verstandene Teile der Legacy-Architektur zurückzuentwickeln und zu dokumentieren, indem Sie Wissen von Entwicklern kombinieren, die verschiedene Teile des Systems verstehen.
- Beziehen Sie praktische Aktivitäten wie kollaboratives Diagrammieren, Architektur-Katas oder geführte Codebasis-Erkundung ein statt passiver Präsentationen.
- Fokussieren Sie jeden Workshop auf ein spezifisches architektonisches Anliegen (z. B. Kopplung zwischen zwei Modulen reduzieren, eine API-Grenze designen, einen Technologiemigrationspfad bewerten), um Diskussionen produktiv zu halten.
- Laden Sie Teilnehmer aus verschiedenen Teams und Erfahrungsstufen ein, um vielfältige Perspektiven sicherzustellen und architektonisches Wissen über die Organisation zu verbreiten.
- Dokumentieren Sie Workshop-Ergebnisse und -Entscheidungen und verfolgen Sie Folgemaßnahmen, um sicherzustellen, dass Workshop-Erkenntnisse sich in tatsächliche Verbesserungen übersetzen.

## Tradeoffs ⇄

> Architektur-Workshops bauen gemeinsames Verständnis auf und treiben architektonische Verbesserung voran, erfordern aber Zeitinvestition und geschickte Moderation.

**Vorteile:**

- Bricht Wissenssilos auf, indem Entwickler zusammengebracht werden, die verschiedene Teile des Legacy-Systems verstehen.
- Baut teamweites architektonisches Bewusstsein auf und verringert das Risiko, dass einzelne Änderungen unbeabsichtigt die Gesamtarchitektur verschlechtern.
- Schafft ein Forum zur Diskussion und Auflösung architektonischer Spannungen, die einzelne Teams nicht allein angehen können.
- Erzeugt Dynamik für Modernisierung, indem dem Team geholfen wird, die Zielarchitektur gemeinsam zu visualisieren und zu planen.

**Kosten und Risiken:**

- Workshops verbrauchen Entwicklungszeit von mehreren Teammitgliedern gleichzeitig, was unter Lieferdruck schwer zu rechtfertigen sein kann.
- Ohne geschickte Moderation können Workshops zu unfokussierten Debatten oder Beschwerdesitzungen entarten, die keine umsetzbaren Ergebnisse produzieren.
- Workshop-Entscheidungen könnten nicht umgesetzt werden, wenn kein Folgemaßnahmen-Mechanismus existiert, um resultierende Arbeitspunkte zu verfolgen und zu priorisieren.
- Teilnehmer ohne ausreichenden Kontext könnten Rauschen statt Signal beitragen, was den Workshop für erfahrene Architekten weniger produktiv macht.

## How It Could Be

> Das folgende Szenario veranschaulicht, wie Architektur-Workshops die Legacy-Modernisierung vorantreiben.

Ein Gesundheitssoftwareunternehmen hielt vierteljährliche Architektur-Workshops ab, bei denen Entwickler aus fünf Teams einen ganzen Tag mit architektonischen Herausforderungen verbrachten. In einem Workshop kartierten die Teams gemeinsam alle Datenflüsse zwischen den 14 Modulen ihres Legacy-Monolithen und entdeckten drei zirkuläre Abhängigkeiten, von denen kein einzelnes Team gewusst hatte. Der Workshop produzierte einen konkreten Plan, diese Zyklen durch die Einführung ereignisbasierter Kommunikation zu durchbrechen, den die Teams im folgenden Quartal umsetzten. In einem anderen Workshop bewerteten die Teams zwei konkurrierende Ansätze zur Migration des Authentifizierungsmoduls und erreichten Konsens über einen Ansatz, den keines der einzelnen Teams in Betracht gezogen hatte. Die Workshops wurden zum primären Ort für teamübergreifende architektonische Abstimmung und wurden für eine 40-prozentige Reduktion teamübergreifender Integrationsprobleme über zwei Jahre verantwortlich gemacht.
