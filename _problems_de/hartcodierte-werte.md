---
title: Hartcodierte Werte
description: Magische Zahlen und feste Strings verringern die Flexibilität und erschweren
  Konfiguration und Anpassung.
category:
- Architecture
- Code
related_problems:
- slug: code-duplication
  similarity: 0.6
- slug: legacy-business-logic-extraction-difficulty
  similarity: 0.6
- slug: difficult-to-understand-code
  similarity: 0.6
- slug: legacy-configuration-management-chaos
  similarity: 0.6
- slug: customization-outside-version-control
  similarity: 0.55
- slug: configuration-chaos
  similarity: 0.55
solutions:
- incremental-refactoring
- secret-management
- static-analysis-and-linting
- abstracted-file-system-access
- externalized-configuration
- platform-independent-configuration-files
- platform-independent-configuration-management
- rule-based-systems
- value-range-definition
- environment-variables-for-configuration
- localization
layout: problem
lang: de
en_slug: hardcoded-values
---

## Description

Hartcodierte Werte sind literale Zahlen, Strings oder andere Konstanten, die direkt in den Quellcode eingebettet sind, statt als konfigurierbare Parameter, Konstanten oder externe Konfiguration definiert zu werden. Diese Praxis verringert die Systemflexibilität, weil sie es erschwert, Verhalten zu ändern, ohne Code zu ändern und neu zu deployen. Das Problem ist besonders problematisch in Systemen, die sich an unterschiedliche Umgebungen anpassen, unterschiedliche Geschäftsregeln handhaben oder sich ändernde Anforderungen über die Zeit berücksichtigen müssen.

## Indicators ⟡

- Code, der unerklärte numerische Literale oder "magische Zahlen" ohne Kontext enthält
- String-Werte wie URLs, Dateipfade oder Meldungen, die direkt in der Geschäftslogik eingebettet sind
- Verschiedene Versionen ähnlichen Codes, die sich nur durch hartcodierte Werte unterscheiden
- Anfragen für "einfache" Konfigurationsänderungen, die Code-Modifikationen erfordern
- Schwierigkeiten beim Einrichten derselben Anwendung in unterschiedlichen Umgebungen
- Geschäftsregeln, die als literale Werte über die Codebasis verstreut sind
- Testdateien, die Produktionscode duplizieren, nur um eingebettete Werte zu ändern

## Symptoms ▲

- [Inkonsistenzen zwischen Deployment-Umgebungen](inkonsistenzen-zwischen-deployment-umgebungen.md)
<br/>  Hartcodierte Werte, die für eine Umgebung korrekt sind, verursachen Fehler, wenn die Anwendung in unterschiedlichen Umgebungen deployt wird.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Einfache Konfigurationsänderungen erfordern Code-Modifikationen, Tests und Neu-Deployment statt nur eine Konfigurationsdatei zu aktualisieren.
- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Code mit hartcodierten Werten bricht leicht, wenn sich Geschäftsregeln, URLs oder andere Parameter ändern.
- [Code-Duplizierung](code-duplizierung.md)
<br/>  Verschiedene Versionen ähnlichen Codes, die sich nur durch hartcodierte Werte unterscheiden, schaffen duplizierte Logik über die gesamte Codebasis.
- [Schwierige Code-Wiederverwendung](schwierige-code-wiederverwendung.md)
<br/>  Code mit eingebetteten literalen Werten kann nicht leicht in unterschiedlichen Kontexten oder Konfigurationen wiederverwendet werden.
- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Was einfache Konfigurationsänderungen sein sollten, werden zu mehrwöchigen Entwicklungsprojekten, die Code-Änderungen und vollständiges Testen erfordern.
- [Inkonsistentes Verhalten](inkonsistentes-verhalten.md)
<br/>  Hartcodierte Werte in verschiedenen Teilen der Codebasis, die eigentlich gleich sein sollten, aber über die Zeit auseinanderdriften, verursachen direkt inkonsistentes Verhalten.

## Causes ▼

- [Termindruck](termindruck.md)
<br/>  Zeitdruck führt dazu, dass Entwickler Werte direkt in den Code einbetten, als schnellster Weg zu einer funktionierenden Lösung.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler ohne Erfahrung mit Konfigurationsmanagement-Mustern greifen standardmäßig darauf zurück, Werte direkt im Quellcode zu hartcodieren.
- [Schlechte Planung](schlechte-planung.md)
<br/>  Das Versäumnis, zukünftige Konfigurationsbedürfnisse zu antizipieren, führt dazu, dass Werte im Code eingebettet statt externalisiert werden.

## Detection Methods ○

- Nutzung statischer Analysewerkzeuge zur Identifikation magischer Zahlen und wiederholter String-Literale
- Code-Reviews, die gezielt nach unerklärten literalen Werten in der Geschäftslogik suchen
- Analyse von Deployment-Prozessen zur Identifikation von Werten, die sich zwischen Umgebungen ändern müssen
- Überprüfung von Konfigurationsänderungsanfragen zur Identifikation von Mustern hartcodierter Abhängigkeiten
- Untersuchung von Testcode auf Workarounds, die aufgrund unflexibler hartcodierter Werte nötig sind
- Befragung von Betriebs- und Geschäftsteams zu Einschränkungen bei der Systemkonfiguration
- Audit der Codebasis auf wiederholte literale Werte, die als Konstanten zentralisiert werden sollten
- Beobachtung der Entwicklungszeit, die für Änderungen aufgewendet wird, die einfache Konfigurationsaktualisierungen sein sollten

## Examples

Eine E-Commerce-Anwendung hat Versandkostenberechnungen, die über die gesamte Codebasis hartcodiert sind, mit Werten wie `if (weight > 50) shippingCost = 15.99` und Timeout-Werten wie `setTimeout(checkStatus, 30000)`. Wenn das Unternehmen Werbeaktionen anbieten, Versandtarife für unterschiedliche Regionen anpassen oder die Performance durch Ändern der Timeout-Werte optimieren möchte, erfordert jede Änderung Code-Modifikationen, Tests und Deployment. Eine besonders problematische Situation entsteht, wenn internationale Kunden unterstützt werden müssen – die hartcodierten USD-Währungssymbole, US-Postleitzahlen-Validierungsmuster und englischen Fehlermeldungen sind über Dutzende Dateien verstreut. Was einfache Geschäftskonfigurationsänderungen sein sollten, werden zu mehrwöchigen Entwicklungsprojekten, und die Unterstützung mehrerer Märkte erfordert die Pflege separater Code-Zweige mit unterschiedlichen hartcodierten Werten.
