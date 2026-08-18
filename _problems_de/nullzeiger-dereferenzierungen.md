---
title: Nullzeiger-Dereferenzierungen
description: Programme versuchen, über Null- oder ungültige Zeiger auf Speicher zuzugreifen,
  was Abstürze und potenzielle Sicherheitslücken verursacht.
category:
- Code
- Security
related_problems:
- slug: buffer-overflow-vulnerabilities
  similarity: 0.65
- slug: stack-overflow-errors
  similarity: 0.6
- slug: inadequate-error-handling
  similarity: 0.5
- slug: memory-leaks
  similarity: 0.5
- slug: integer-overflow-underflow
  similarity: 0.5
solutions:
- static-analysis-and-linting
- fuzz-testing
- negative-testing
- design-by-contract
- input-validation
- code-reviews
- property-based-testing
- error-handling
layout: problem
lang: de
en_slug: null-pointer-dereferences
---

## Description

Nullzeiger-Dereferenzierungen treten auf, wenn ein Programm versucht, über einen Zeiger auf Speicher zuzugreifen, der null ist (auf Speicheradresse 0 zeigt) oder eine ungültige Speicheradresse enthält. Dies ist einer der häufigsten Laufzeitfehler in der Systemprogrammierung und kann sofortige Anwendungsabstürze, Datenkorruption oder Sicherheitslücken verursachen. Der Fehler äußert sich typischerweise als Segmentation Fault, Zugriffsverletzung oder Nullzeiger-Ausnahme, abhängig von Programmiersprache und Laufzeitumgebung.

## Indicators ⟡

- Die Anwendung stürzt mit Segmentation Faults, Zugriffsverletzungen oder Nullzeiger-Ausnahmen ab
- Abstürze treten beim Zugriff auf Objektmethoden oder -eigenschaften bei potenziell null Referenzen auf
- Debugging zeigt null oder ungültige Zeigerwerte am Absturzpunkt
- Abstürze treten inkonsistent auf, abhängig vom Programmzustand oder den Eingabebedingungen
- Stack-Traces verweisen auf Speicherzugriffsoperationen bei Nullzeigern

## Symptoms ▲

- [Systemausfälle](systemausfaelle.md)
<br/>  Nullzeiger-Dereferenzierungen verursachen Anwendungsabstürze und Segmentation Faults, was direkt zu Serviceunterbrechungen und Systemausfällen führt.
- [Erhöhte Fehlerraten](erhoehte-fehlerraten.md)
<br/>  Nullzeiger-Ausnahmen äußern sich als Laufzeitfehler, die die Gesamtfehlerrate der Anwendung erhöhen.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Nullzeiger-Dereferenzierungen, die inkonsistent abhängig vom Programmzustand auftreten, sind notorisch schwer zu reproduzieren und zu debuggen.
- [Unvorhersehbares Systemverhalten](unvorhersehbares-systemverhalten.md)
<br/>  Nullzeiger-Dereferenzierungen verursachen Abstürze, die inkonsistent abhängig von Eingabebedingungen und Programmzustand auftreten, was das Systemverhalten unvorhersehbar macht.
- [Stille Datenkorruption](stille-datenkorruption.md)
<br/>  In manchen Fällen können Nullzeiger-Dereferenzierungen benachbarten Speicher korrumpieren, statt sofort abzustürzen, was zu stiller Datenkorruption führt.

## Causes ▼

- [Unzureichende Fehlerbehandlung](unzureichende-fehlerbehandlung.md)
<br/>  Das Versäumnis, Rückgabewerte zu prüfen und Zeiger vor der Nutzung zu validieren, ist eine direkte Ursache für Nullzeiger-Dereferenzierungen.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Ohne automatisierte Tests, die Nullzeiger-Bedingungen ausüben, bleiben diese Defekte unentdeckt, bis sie Produktionsabstürze verursachen.
- [Schlechte Testabdeckung](schlechte-testabdeckung.md)
<br/>  Unzureichende Testabdeckung bedeutet, dass Nullzeiger-Randfälle nicht getestet werden, was Dereferenzierungsfehler in die Produktion gelangen lässt.

## Detection Methods ○

- **Statische Analysewerkzeuge:** Nutzung von Werkzeugen, die potenzielle Nullzeiger-Dereferenzierungspfade im Quellcode identifizieren können
- **Laufzeitprüfung:** Nutzung von Laufzeitwerkzeugen wie AddressSanitizer oder Valgrind zur Erkennung von Nullzeigerzugriffen
- **Null-Prüfungsanalyse:** Überprüfung des Codes auf ordentliche Null-Prüfung vor Zeiger-Dereferenzierung
- **Überprüfung der Ausnahmebehandlung:** Analyse der Ausnahmebehandlung zur Sicherstellung, dass Zeiger validiert werden
- **Unit-Testing:** Erstellung von Tests, die spezifisch Nullzeiger-Bedingungen ausüben
- **Code-Review:** Manuelle Überprüfung mit Fokus auf Zeigerinitialisierung und Validierungsmuster

## Examples

Ein C-Programm alloziert Speicher mittels malloc, prüft aber nicht, ob die Allokation erfolgreich war. Wenn der Speicher erschöpft ist, gibt malloc NULL zurück, aber das Programm nutzt weiterhin den Nullzeiger, als zeige er auf gültigen Speicher, was einen Segmentation Fault verursacht, wenn es versucht, Daten zu schreiben. Ein weiteres Beispiel betrifft eine Java-Anwendung, die ein Objekt aus einer Sammlung abruft, die leer sein könnte. Ohne zu prüfen, ob das zurückgegebene Objekt null ist, ruft der Code sofort eine Methode auf dem Objekt auf, was zu einer NullPointerException führt, die den Anwendungs-Thread zum Absturz bringt und das System möglicherweise in einem inkonsistenten Zustand zurücklässt.
