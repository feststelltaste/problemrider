---
title: Lange Build- und Testzeiten
description: Eine Situation, in der es lange dauert, ein System zu bauen und zu testen.
category:
- Process
related_problems:
- slug: long-release-cycles
  similarity: 0.7
- slug: extended-cycle-times
  similarity: 0.7
- slug: inefficient-development-environment
  similarity: 0.65
- slug: extended-research-time
  similarity: 0.65
- slug: difficult-developer-onboarding
  similarity: 0.6
- slug: work-queue-buildup
  similarity: 0.6
solutions:
- ci-cd-pipeline
- continuous-integration
- continuous-integration-and-delivery
- cross-platform-build-scripts
- cross-platform-build-tools
- parallelization
- pipelining
- platform-independent-build-pipelines
- fast-feedback-loops
layout: problem
lang: de
en_slug: long-build-and-test-times
---

## Description
Lange Build- und Testzeiten sind eine Situation, in der es lange dauert, ein System zu bauen und zu testen. Dies ist ein häufiges Problem in großen, monolithischen Architekturen, bei denen das gesamte System auf einmal gebaut und getestet werden muss. Lange Build- und Testzeiten können zu einer Verlangsamung der Entwicklungsgeschwindigkeit führen, und sie können auch eine erhebliche Frustrationsquelle für Entwickler sein.

## Indicators ⟡
- Es dauert lange, Feedback zu einer Änderung zu erhalten.
- Entwickler sind oft blockiert und warten auf den Abschluss des Builds.
- Der Build ist häufig defekt.
- Entwickler sind nicht in der Lage, die Tests auf ihren lokalen Rechnern auszuführen.

## Symptoms ▲

- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Entwickler, die auf Builds und Tests warten, können nicht schnell iterieren, was den Entwicklungsdurchsatz des Teams direkt verringert.
- [Verringerte Häufigkeit von Code-Einreichungen](verringerte-haeufigkeit-von-code-einreichungen.md)
<br/>  Wenn Builds zu lange dauern, bündeln Entwickler Änderungen, um Wartezeiten zu vermeiden, und reichen seltener ein.
- [Langlebige Feature-Branches](langlebige-feature-branches.md)
<br/>  Langsames Feedback von Builds entmutigt häufige Integration, was dazu führt, dass Branches länger leben, bevor sie gemerged werden.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Ständiges Warten auf langsame Builds ist demoralisierend und unterbricht den Entwickler-Flow, was zu Frustration führt.
- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Entwickler überspringen das Ausführen vollständiger Testsuiten lokal aufgrund langer Zeiten, was zu mehr Defekten führt, die geteilte Branches erreichen.
- [Overhead durch Kontextwechsel](overhead-durch-kontextwechsel.md)
<br/>  Während des Wartens auf Builds wechseln Entwickler zu anderen Aufgaben, verlieren mentalen Kontext und verringern ihre Effektivität.

## Causes ▼

- [Einschränkungen durch monolithische Architektur](einschraenkungen-durch-monolithische-architektur.md)
<br/>  Monolithische Systeme erfordern, dass die gesamte Anwendung gemeinsam gebaut und getestet wird, was Build-Zeiten mit der Systemgröße wachsen lässt.
- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Eng gekoppelte Komponenten verhindern inkrementelle Builds und erfordern vollständige Neukompilierung und Tests für jede Änderung.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Schlechte Modulgrenzen bedeuten, dass Änderungen über die Codebasis kaskadieren, was umfangreiches Neubauen und erneutes Testen erfordert.
- [Unkontrolliertes Wachstum der Codebasis](unkontrolliertes-wachstum-der-codebasis.md)
<br/>  Eine stetig wachsende Codebasis ohne Modularisierung führt natürlicherweise zu längeren Kompilierungs- und Testausführungszeiten.
- [Unzureichende Testinfrastruktur](unzureichende-testinfrastruktur.md)
<br/>  Fehlende ordentliche Testinfrastruktur wie parallele Testausführung oder Test-Caching macht Testläufe unnötig langsam.

## Detection Methods ○
- **Überwachung der Build- und Testzeiten:** Überwachung der Build- und Testzeiten zur Identifikation, welche Teile des Builds am langsamsten sind.
- **Entwicklerbefragungen:** Befragung von Entwicklern, ob sie das Gefühl haben, schnelles Feedback zu ihren Änderungen zu erhalten.
- **Analyse der Build- und Test-Logs:** Analyse der Build- und Test-Logs zur Identifikation von Fehlern und Warnungen.

## Examples
Ein Unternehmen hat eine große, monolithische E-Commerce-Anwendung. Es dauert über eine Stunde, die Anwendung zu bauen und zu testen. Die Entwickler sind oft blockiert und warten auf den Abschluss des Builds. Der Build ist häufig defekt, und es kann Stunden dauern, ihn zu beheben. Die Entwickler sind nicht in der Lage, alle Tests auf ihren lokalen Rechnern auszuführen, sodass sie kein vollständiges Bild von der Qualität ihres Codes erhalten können. Infolgedessen ist die Entwicklungsgeschwindigkeit langsam, und die Codequalität ist schlecht.
