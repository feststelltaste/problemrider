---
title: ABI-Kompatibilitätsprobleme
description: Inkompatibilitäten der Binärschnittstelle (ABI) zwischen verschiedenen
  Versionen von Bibliotheken oder Systemkomponenten führen zu Laufzeitfehlern oder
  undefiniertem Verhalten.
category:
- Code
- Dependencies
- Testing
related_problems:
- slug: dependency-version-conflicts
  similarity: 0.6
- slug: api-versioning-conflicts
  similarity: 0.6
- slug: legacy-api-versioning-nightmare
  similarity: 0.55
- slug: breaking-changes
  similarity: 0.55
- slug: poor-interfaces-between-applications
  similarity: 0.55
- slug: deployment-environment-inconsistencies
  similarity: 0.5
solutions:
- compatibility-testing
- compatibility-matrix
- semantic-versioning
- abstraction-layers
- cross-version-testing
- dependency-pinning
- backward-compatibility
- contract-testing
- interoperability-tests
- api-versioning-strategy
- deprecation-strategy
layout: problem
lang: de
en_slug: abi-compatibility-issues
---

## Description

ABI (Application Binary Interface)-Kompatibilitätsprobleme entstehen, wenn Anwendungen, die gegen eine Version einer Bibliothek oder Systemkomponente kompiliert wurden, mit einer anderen Version verwendet werden, die inkompatible Binärschnittstellen aufweist. Dies kann Abstürze, Speicherkorruption, fehlerhaftes Verhalten oder Ladefehler verursachen, da die Anwendung andere Funktionssignaturen, Datenlayouts oder Aufrufkonventionen erwartet, als die Laufzeitbibliothek bereitstellt.

## Indicators ⟡

- Anwendungen stürzen sofort beim Start ab oder beim Aufruf bestimmter Bibliotheksfunktionen
- Funktionen liefern unerwartete Werte zurück oder verhalten sich anders als erwartet
- Speicherkorruption oder Segmentation Faults treten im Bibliotheksinteraktionscode auf
- Dynamisches Linken schlägt mit Symbolauflösungsfehlern fehl
- Anwendungen funktionieren in der Entwicklung, scheitern aber in der Produktion mit unterschiedlichen Bibliotheksversionen

## Symptoms ▲

- [Kaskadierende Ausfälle](kaskadierende-ausfaelle.md)
<br/>  Wenn Komponenten eng gekoppelt sind oder keine Fehlerisolation aufweisen, kann ein durch ABI ausgelöster Absturz in einer Komponente sich über Abhängige fortpflanzen und kaskadierende Ausfälle im gesamten System auslösen.
- [Integrationsschwierigkeiten](integrationsschwierigkeiten.md)
<br/>  Wenn die Integration kompilierte oder native Komponenten betrifft (z. B. Shared Libraries, Plugins), machen Binärschnittstellen-Unstimmigkeiten zwischen Bibliotheksversionen die Integration extrem schwierig.
- [Erhöhte Fehlerraten](erhoehte-fehlerraten.md)
<br/>  Laufzeitfehler durch ABI-Unstimmigkeiten führen zu erhöhten Fehlerraten, da Funktionsaufrufe unerwartete Werte zurückgeben oder abstürzen.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  ABI-Probleme verursachen subtile Speicherkorruption und undefiniertes Verhalten, die extrem schwer zu diagnostizieren und zu debuggen sind.

## Causes ▼

- [Versionskonflikte bei Abhängigkeiten](versionskonflikte-bei-abhaengigkeiten.md)
<br/>  Unterschiedliche Komponenten, die von unterschiedlichen Versionen derselben Bibliothek abhängen, sind eine Hauptursache für ABI-Inkompatibilitäten.
- [Breaking Changes](breaking-changes.md)
<br/>  Wenn Breaking Changes die exportierten Funktionssignaturen oder Datenlayouts einer kompilierten Bibliothek ohne ordentliche Versionierung ändern, verursachen sie direkt ABI-Kompatibilitätsprobleme; Änderungen an nicht-binären Schnittstellen (z. B. REST-APIs) verursachen stattdessen API-Level-Inkompatibilitäten statt ABI-Problemen.
- [Inkonsistenzen zwischen Deployment-Umgebungen](inkonsistenzen-zwischen-deployment-umgebungen.md)
<br/>  Wenn unterschiedliche Umgebungen unterschiedliche Versionen von Shared Libraries installieren, kann Code, der gegen eine ABI kompiliert wurde, zur Laufzeit eine inkompatible Bibliotheksversion laden, was zu Fehlern führt, die erst außerhalb der Entwicklung auftreten.

## Detection Methods ○

- **Binäranalyse-Werkzeuge:** Nutzung von Werkzeugen zum Vergleich der ABI-Kompatibilität zwischen Bibliotheksversionen
- **Symbolverifikation:** Prüfung, ob erwartete Symbole existieren und korrekte Signaturen haben
- **Laufzeittests:** Testen von Anwendungen mit verschiedenen Bibliotheksversionen zur Identifikation von Inkompatibilitäten
- **Linking-Analyse:** Analyse des Linking-Verhaltens und der Symbolauflösung beim Anwendungsstart
- **Speicherlayout-Verifikation:** Überprüfung, dass Datenstrukturlayouts zwischen Kompilierungs- und Laufzeitversionen übereinstimmen
- **Kompatibilitätstest-Suiten:** Nutzung automatisierter Tests zur Verifikation der ABI-Kompatibilität über Versionen hinweg

## Examples

Eine Anwendung, die gegen Version 1.0 einer Grafikbibliothek kompiliert wurde, erwartet eine Color-Struktur mit drei Integer-Feldern (RGB), aber Version 2.0 änderte die Struktur auf vier Felder (RGBA). Wenn die Anwendung mit der neuen Bibliothek läuft, korrumpiert sie Speicher, indem sie über die erwartete Strukturgrenze hinaus schreibt, was Abstürze und unvorhersehbares Verhalten verursacht. Ein weiteres Beispiel betrifft eine Netzwerkbibliothek, die eine Funktionssignatur von `send_data(char* data, int length)` zu `send_data(const char* data, size_t length)` zwischen Versionen änderte. Anwendungen, die gegen die alte Version kompiliert wurden, übergeben falsche Parametertypen, was zu Datenkorruption oder Abstürzen führt, wenn der Size-Parameter falsch interpretiert wird.
