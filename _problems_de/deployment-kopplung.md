---
title: Deployment-Kopplung
description: Eine Situation, in der mehrere Komponenten oder Services gemeinsam
  deployt werden müssen, selbst wenn sich nur eine davon geändert hat.
category:
- Architecture
- Operations
related_problems:
- slug: tight-coupling-issues
  similarity: 0.7
- slug: shared-dependencies
  similarity: 0.65
- slug: complex-deployment-process
  similarity: 0.65
- slug: high-coupling-low-cohesion
  similarity: 0.65
- slug: deployment-risk
  similarity: 0.6
- slug: ripple-effect-of-changes
  similarity: 0.6
solutions:
- ci-cd-pipeline
- event-driven-architecture
- event-driven-integration
- microservices
- microservices-architecture
- modulith
- rolling-updates
- trunk-based-development
layout: problem
lang: de
en_slug: deployment-coupling
---

## Description
Deployment-Kopplung ist eine Situation, in der mehrere Komponenten oder Services gemeinsam deployt werden müssen, selbst wenn sich nur eine davon geändert hat. Dies ist ein verbreitetes Problem in monolithischen Architekturen, bei denen alle Komponenten eng gekoppelt und als eine einzige Einheit deployt werden. Deployment-Kopplung kann zu langen Release-Zyklen, großen und riskanten Releases und erheblicher Angst im Entwicklungsteam führen.

## Indicators ⟡
- Eine kleine Änderung an einer Komponente erfordert das Neu-Deployment des gesamten Systems.
- Es ist nicht möglich, unterschiedliche Komponenten des Systems unabhängig zu deployen.
- Der Deployment-Prozess ist komplex und fehleranfällig.
- Das Entwicklungsteam hat Angst, Änderungen am System vorzunehmen, aus Furcht, etwas zu brechen.

## Symptoms ▲

- [Große, riskante Releases](grosse-riskante-releases.md)
<br/>  Wenn Komponenten gemeinsam deployt werden müssen, häufen Releases viele Änderungen über mehrere Komponenten hinweg an, was Größe und Risiko erhöht.
- [Lange Release-Zyklen](lange-release-zyklen.md)
<br/>  Die Koordination von Deployments über gekoppelte Komponenten hinweg verlängert die Zeit zwischen Releases.
- [Deployment-Risiko](deployment-risiko.md)
<br/>  Das gleichzeitige Deployen mehrerer Komponenten erhöht die Wahrscheinlichkeit von Fehlschlägen und macht Rollbacks komplexer.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Einzelne Features werden zurückgehalten, bis alle gekoppelten Komponenten für das Deployment bereit sind.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Die Komplexität und das Risiko gekoppelter Deployments macht Teams zurückhaltend, Änderungen vorzunehmen.
- [Release-Angst](release-angst.md)
<br/>  Teams erleben Angst rund um Deployments, weil gekoppelte Releases mehr bewegliche Teile haben, die fehlschlagen können.

## Causes ▼

- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Komponenten, die auf Code-Ebene eng gekoppelt sind, erfordern notwendigerweise koordiniertes Deployment.
- [Gemeinsam genutzte Datenbank](gemeinsam-genutzte-datenbank.md)
<br/>  Komponenten, die eine Datenbank gemeinsam nutzen, müssen gemeinsam deployt werden, wenn Schemaänderungen mehrere Services betreffen.
- [Einschränkungen durch monolithische Architektur](einschraenkungen-durch-monolithische-architektur.md)
<br/>  Monolithische Architekturen bündeln inhärent alle Komponenten in eine einzige deploybare Einheit.
- [Gemeinsam genutzte Abhängigkeiten](gemeinsam-genutzte-abhaengigkeiten.md)
<br/>  Gemeinsam genutzte Bibliotheken oder Services schaffen Deployment-Kopplung, wenn Aktualisierungen an der gemeinsamen Komponente koordinierte Releases erfordern.
- [API-Versionierungskonflikte](api-versionierungskonflikte.md)
<br/>  Wenn Services keine Backward-Compatibility-Schicht oder kein Gateway haben, um Versionsunterschiede abzufangen, können ungelöste API-Versionskonflikte Teams dazu zwingen, Deployments über mehrere Services hinweg zu koordinieren.

## Detection Methods ○
- **Deployment-Prozess-Mapping:** Abbildung der Schritte im Deployment-Prozess zur Identifikation von Engpässen und Verbesserungsbereichen.
- **Komponentenabhängigkeitsanalyse:** Analyse der Abhängigkeiten zwischen Komponenten zur Identifikation, welche Komponenten unabhängig deployt werden können.
- **Entwickler-Umfragen:** Befragung von Entwicklern, ob sie das Gefühl haben, ihre Änderungen schnell und sicher deployen zu können.

## Examples
Ein Unternehmen hat eine große, monolithische E-Commerce-Anwendung. Die Anwendung besteht aus mehreren unterschiedlichen Komponenten, einschließlich eines Produktkatalogs, eines Warenkorbs und eines Zahlungs-Gateways. Die Komponenten sind alle eng gekoppelt und werden als eine einzige Einheit deployt. Wenn das Entwicklungsteam eine Änderung am Produktkatalog vornehmen möchte, muss es die gesamte Anwendung neu deployen. Dies ist ein zeitaufwendiger und riskanter Prozess und führt oft zu Problemen. Infolgedessen kann das Unternehmen nur einmal im Monat neue Software veröffentlichen.
