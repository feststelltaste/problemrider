---
title: Strategisches Löschen von Code
description: Gezielte Entfernung überflüssigen oder veralteten Codes zur
  Reduzierung der Codebasis.
category:
- Code
problems:
- uncontrolled-codebase-growth
- code-duplication
- difficult-code-comprehension
- high-maintenance-costs
- feature-bloat
- increased-cognitive-load
- accumulation-of-workarounds
- copy-paste-programming
- maintenance-cost-increase
- custom-report-sprawl
- low-code-customization-sprawl
- reimplemented-standard-functionality
layout: solution
lang: de
en_slug: strategic-code-deletion
related_solutions:
- slug: tree-shaking
  similarity: 0.75
- slug: deprecation-strategy
  similarity: 0.75
- slug: facades
  similarity: 0.7
- slug: clean-code
  similarity: 0.7
- slug: data-deduplication
  similarity: 0.7
- slug: pattern-language
  similarity: 0.7
---

## Description

Strategisches Löschen von Code ist die bewusste, laufende Entfernung von Code, der keinem Zweck mehr dient — unerreichbare Methoden, veraltete Feature-Flag-Zweige, obsolete Tests, auskommentierte Fragmente, ganze Module für eingestellte Funktionalität —, identifiziert durch statische Analyse, Versionskontrollhistorie und Teamwissen, statt opportunistisch entfernt oder aus Vorsicht belassen zu werden. Legacy-Codebasen wachsen jahrelang monoton, weil Hinzufügen immer einfacher und weniger riskant ist als Wegnehmen: Niemand will derjenige sein, der Code löscht, der sich als bedeutsam herausstellt, sodass sich toter Code ansammelt und jeder Entwickler die kognitive Last erbt, Funktionalität zu lesen und potenziell zu pflegen, die nicht mehr ausgeführt wird. Diese Ansammlung umzukehren reduziert direkt die Größe des Systems, das verstanden, kompiliert und getestet werden muss, was Build-Zeiten verkürzt und die Codebasis zugänglicher macht, besonders für neue Entwickler, die versuchen, ein mentales Modell davon aufzubauen, was das System tatsächlich tut. Dies unterscheidet sich von allgemeinem Refactoring darin, dass sein Ergebnis negativ ist — das Ziel ist ein kleineres System, kein anders strukturiertes —, und es hängt von der Zuversicht ab, dass entfernter Code genuin unerreichbar ist, was entweder starke Testabdeckung oder sorgfältige Analyse erfordert, um sich davor zu schützen, versehentlich etwas zu löschen, das über Reflection, dynamischen Dispatch oder Konfiguration statt direkten Aufruf ausgelöst wird. Als regelmäßige, inkrementelle Wartungsaktivität behandelt statt als einmaliges Bereinigungsprojekt, neigt es auch dazu, Fehler zutage zu bringen, die still hinter toten Codepfaden verborgen waren.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Nutzen Sie statische Analysewerkzeuge und IDE-Features, um toten Code zu identifizieren: unerreichbare Methoden, ungenutzte Imports und nicht aufgerufene Funktionen
- Prüfen Sie die Versionskontrollhistorie, um Code zu finden, der lange nicht modifiziert oder ausgeführt wurde
- Entfernen Sie Feature-Flags und ihre zugehörigen Codepfade, sobald Features dauerhaft aktiviert oder deaktiviert sind
- Löschen Sie auskommentierte Codeblöcke; die Versionskontrolle bewahrt die Historie, falls der Code jemals wieder benötigt wird
- Entfernen Sie obsoleten Testcode, der gelöschte oder veraltete Funktionalität testet
- Koordinieren Sie Löschungen mit dem Team, um zu vermeiden, Code zu entfernen, den jemand reaktivieren möchte
- Machen Sie Code-Löschung zu einer regelmäßigen Wartungsaktivität statt einem einmaligen Ereignis

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Reduziert kognitive Last, indem die Menge an Code verkleinert wird, die Entwickler verstehen müssen
- Senkt Wartungskosten durch Beseitigung von Code, der noch kompilieren und Tests bestehen muss
- Verbessert Build- und Testzeiten durch Entfernung unnötiger Kompilierungs- und Testziele
- Macht die Codebasis für neue Entwickler zugänglicher

**Kosten und Risiken:**
- Risiko, Code zu löschen, der über Reflection, dynamischen Dispatch oder konfigurationsgesteuerten Aufruf genutzt wird
- Erfordert gute Testabdeckung, um zu validieren, dass nach der Löschung nichts kaputtgeht
- Entwickler könnten sich gegen das Löschen von Code sträuben, in dessen Schreiben sie Aufwand investiert haben
- In Legacy-Systemen kann es schwer sein zu bestimmen, ob Code wirklich ungenutzt ist

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Das Legacy-Content-Management-System eines Medienunternehmens war über 15 Jahre auf über zwei Millionen Zeilen Code angewachsen. Ein statischer Analyse-Scan offenbarte, dass ungefähr 18 % der Codebasis unerreichbarer toter Code war, einschließlich ganzer Module für eingestellte Produktlinien. Das Team führte über drei Sprints eine systematische Löschungsanstrengung durch und entfernte den toten Code in sorgfältig überprüften Paketen. Die Build-Zeiten sanken um 12 %, die Testsuite lief merklich schneller, und neue Entwickler berichteten, dass die Navigation durch die Codebasis erheblich weniger überwältigend wurde. Das Team entdeckte auch mehrere Fehler, die hinter toten Codepfaden versteckt waren und falsches Verhalten verschleiert hatten.
