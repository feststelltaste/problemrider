---
title: Versionskontrolle für Kompatibilität
description: Nachverfolgung und Verwaltung kompatibilitätsrelevanter
  Änderungen über parallele Versionen hinweg.
category:
- Process
- Dependencies
problems:
- api-versioning-conflicts
- breaking-changes
- dependency-version-conflicts
- configuration-drift
- no-formal-change-control-process
- change-management-chaos
- customization-outside-version-control
layout: solution
lang: de
en_slug: version-control
related_solutions:
- slug: semantic-versioning
  similarity: 0.75
- slug: versioning-scheme
  similarity: 0.75
- slug: compatibility-governance
  similarity: 0.7
- slug: compatibility-as-error
  similarity: 0.7
- slug: backward-compatibility
  similarity: 0.7
- slug: api-versioning-strategy
  similarity: 0.7
---

## Description

Versionskontrolle für Kompatibilität ist die Praxis, Änderungen bewusst zu verfolgen, zu verzweigen und zu regeln, die beeinflussen, wie verschiedene Konsumenten einer API, Bibliothek oder eines Datenformats mit ihr interoperieren können, sodass kompatibilitätsrelevante Entscheidungen explizit getroffen werden, statt als Nebeneffekt dessen zu entstehen, was die aktuellen Maintainer zufällig ändern. Dies bedeutet typischerweise, parallele unterstützte Versionen für ein definiertes Deprecation-Fenster zu pflegen, eine Kompatibilitätsmatrix zu dokumentieren, welche Versionen zusammenarbeiten, und automatisierte Kompatibilitätstests über die noch aktiv genutzten Kombinationen hinweg auszuführen. Es adressiert einen spezifischen Fehlermodus, der um Legacy-Integrationen herum häufig ist: Ein zentrales System bedient viele Konsumenten, die zu unterschiedlichen Zeiten gebaut wurden und auf unterschiedlichen Upgrade-Zyklen sind, und ohne bewusste Versionierungsdisziplin riskiert jede Änderung, die zum Nutzen eines Konsumenten gemacht wird, still einen anderen zu brechen, den niemand zu prüfen dachte. Die Praxis ist es, was einer Legacy-Plattform erlaubt, sich weiterzuentwickeln — Sicherheitsfixes anzuwenden, Fähigkeiten hinzuzufügen —, ohne jede abhängige Integration zu zwingen, im Gleichschritt zu aktualisieren, was selten realistisch ist, wenn manche Konsumenten von externen Parteien mit ihren eigenen Release-Zeitplänen gepflegt werden. Die Kosten sind echt: die Pflege mehrerer Live-Versionen und das Zurückportieren von Fixes über sie hinweg ist genuin mehr Arbeit als die Pflege einer, weshalb die Praxis eine feste Deprecation-Richtlinie mit der Unterstützung paralleler Versionen paart, sodass die Last der Unterstützung alter Versionen sich nicht einfach unbegrenzt ansammelt.

## How to Apply ◆

- Pflegen Sie parallele Versionsbranches für Legacy-APIs und -Bibliotheken, die Konsumenten auf unterschiedlichen Upgrade-Zeitplänen haben.
- Etablieren Sie eine Kompatibilitätsmatrix, die dokumentiert, welche Versionen von Diensten und Bibliotheken miteinander kompatibel sind.
- Nutzen Sie Branching-Strategien, die kompatibilitätskritische Änderungen von internen Verbesserungen trennen.
- Automatisieren Sie Kompatibilitätstests über unterstützte Versionskombinationen in der CI-Pipeline.
- Definieren Sie eine klare Deprecation-Richtlinie mit Zeitplänen, sodass Konsumenten wissen, wann ältere Versionen ausgemustert werden.
- Kennzeichnen Sie Releases mit Kompatibilitätsmetadaten und veröffentlichen Sie Release Notes, die Breaking Changes hervorheben.

## Tradeoffs ⇄

**Vorteile:**
- Ermöglicht Konsumenten, nach eigenem Zeitplan zu aktualisieren, ohne zu Breaking Changes gezwungen zu werden.
- Bietet klare Sichtbarkeit darüber, welche Versionen wie lange unterstützt werden.
- Reduziert das Risiko unbeabsichtigter Brüche, indem kompatibilitätsrelevante Änderungen isoliert werden.
- Unterstützt phasierte Migrationsstrategien, die in der Legacy-Modernisierung üblich sind.

**Kosten:**
- Die Pflege mehrerer paralleler Versionen erhöht die Entwicklungs- und Testlast.
- Das Zurückportieren von Fixes über Versionen hinweg ist zeitaufwändig und fehleranfällig.
- Langlebige parallele Versionen können zu Divergenz führen, die zunehmend schwer zu verwalten wird.
- Erfordert Governance, um Deprecation-Zeitpläne durchzusetzen und Versionsproliferation zu verhindern.

## How It Could Be

Eine Legacy-Zahlungsverarbeitungsplattform bietet APIs, die von Dutzenden von Händlerintegrationen konsumiert werden, jede auf unterschiedlichen Upgrade-Zyklen. Das Team übernimmt eine Versionskontrollstrategie, bei der zwei Haupt-API-Versionen gleichzeitig unterstützt werden, mit einem zwölfmonatigen Deprecation-Fenster. Jede Version hat ihren eigenen Branch, und die CI-Pipeline führt Kompatibilitätstests gegen beide durch. Wenn ein Sicherheitsfix benötigt wird, wird er auf beide unterstützten Versionen angewendet. Händler erhalten Deprecation-Benachrichtigungen mit Migrationsleitfäden sechs Monate, bevor eine alte Version ausgemustert wird. Dieser strukturierte Ansatz ersetzt die vorherige Ad-hoc-Praxis, bei der Breaking Changes ohne Warnung deployt wurden, was Integrationsfehler für Händler verursachte, die nicht sofort aktualisieren konnten.
