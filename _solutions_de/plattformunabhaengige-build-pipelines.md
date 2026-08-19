---
title: Plattformunabhängige Build-Pipelines
description: Umsetzung von CI/CD-Pipelines, die auf unterschiedlichen
  Build-Servern laufen.
category:
- Operations
- Process
problems:
- vendor-lock-in
- technology-lock-in
- complex-deployment-process
- manual-deployment-processes
- deployment-environment-inconsistencies
- long-build-and-test-times
- inefficient-development-environment
layout: solution
lang: de
en_slug: platform-independent-build-pipelines
related_solutions:
- slug: cross-platform-build-tools
  similarity: 0.8
- slug: cross-platform-build-scripts
  similarity: 0.75
- slug: platform-independence
  similarity: 0.75
- slug: platform-independent-scripting-languages
  similarity: 0.7
- slug: continuous-integration-and-delivery
  similarity: 0.7
- slug: platform-independent-programming-languages
  similarity: 0.7
---

## Description

Eine plattformunabhängige Build-Pipeline definiert ihre Schritte in einem portablen Format — einem Makefile, Shell-Skripten oder containerisierten Build-Images —, das jeder CI-Server ausführen kann, statt die Build-Logik in den proprietären Plugins und der Pipeline-Syntax eines spezifischen Anbieters zu codieren. Legacy-Build-Systeme driften über Jahre inkrementeller Konfiguration häufig in tiefe Kopplung an eine CI-Plattform, bis zu dem Punkt, an dem ein Anbieterwechsel aussieht, als erfordere er das Neuschreiben jeder Pipeline von Grund auf. Die tatsächliche Build-Logik in portablen, versionskontrollierten Skripten zu halten und die CI-spezifische Konfiguration auf einen dünnen Wrapper darum zu reduzieren, bedeutet, dass eine Anbietermigration zu einer Frage des Neuschreibens dieser dünnen Schicht wird, nicht der Substanz des Builds — auf Kosten dessen, einige anbieterspezifische Annehmlichkeiten wie natives Caching oder Matrix-Builds aufzugeben, die die portable Schicht nicht vollständig replizieren kann.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Definieren Sie Build-Schritte in einem plattformagnostischen Format (z. B. Makefile, Shell-Skripte oder containerisierte Build-Images), das jeder CI-Server aufrufen kann
- Vermeiden Sie die Nutzung CI-server-spezifischer Features oder proprietärer Plugins für die Kern-Build-Logik
- Verwenden Sie containerbasierte Build-Agenten, sodass die Build-Umgebung unabhängig von der CI-Plattform reproduzierbar ist
- Speichern Sie Pipeline-Definitionen als Code im Repository neben dem Anwendungsquellcode
- Abstrahieren Sie umgebungsspezifische Variablen durch eine Konfigurationsschicht, statt sie in Pipeline-Definitionen einzubetten
- Testen Sie die Pipeline periodisch auf mindestens zwei verschiedenen CI-Plattformen, um Portabilität zu verifizieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Vermeidet Lock-in bei einem spezifischen CI/CD-Anbieter, was den Anbieterwechsel erleichtert
- Stellt Build-Reproduzierbarkeit über Entwicklerrechner, CI-Server und Produktionsumgebungen sicher
- Vereinfacht das Onboarding, da Entwickler dieselben Build-Schritte lokal ausführen können
- Reduziert Risiko, wenn ein CI-Anbieter Preise oder Features ändert oder den Dienst einstellt

**Kosten und Risiken:**
- Die Beschränkung auf plattformagnostische Features bedeutet, auf anbieterspezifische Optimierungen wie natives Caching zu verzichten
- Die Aufrechterhaltung von Portabilität über CI-Plattformen hinweg fügt Test- und Validierungsaufwand hinzu
- Containerbasierte Builds können zusätzliche Startlatenz gegenüber nativen Agenten einführen
- Manche fortgeschrittenen Pipeline-Features wie Matrix-Builds oder Freigabe-Gates unterscheiden sich erheblich zwischen Plattformen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein mittelgroßes Softwareunternehmen hatte seine gesamte Build-Pipeline tief in Jenkins eingebettet, mit über 200 Groovy-basierten Pipeline-Skripten, die Jenkins-spezifische Plugins nutzten. Als die Jenkins-Wartung zu einer erheblichen betrieblichen Belastung wurde, schien die Migration zu GitHub Actions das Neuschreiben jeder Pipeline zu erfordern. Das Team refaktorierte, indem es die Kern-Build-Logik in Makefiles und Docker-basierte Build-Images extrahierte, mit dünnen CI-spezifischen Wrappern, die einfach diese portablen Schritte aufriefen. Die Migration zu GitHub Actions dauerte drei Wochen statt der geschätzten drei Monate, und sie behielten die Fähigkeit, bei Bedarf auf Jenkins zurückzufallen.
