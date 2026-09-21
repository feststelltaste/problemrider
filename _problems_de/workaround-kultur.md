---
title: Workaround-Kultur
description: Teams implementieren zunehmend komplexe Workarounds statt Grundursachen
  zu beheben, was Schichten technischer Schulden schafft.
category:
- Code
- Culture
- Process
related_problems:
- slug: accumulation-of-workarounds
  similarity: 0.9
- slug: increased-technical-shortcuts
  similarity: 0.75
- slug: high-technical-debt
  similarity: 0.7
- slug: complex-implementation-paths
  similarity: 0.65
- slug: delayed-issue-resolution
  similarity: 0.65
- slug: refactoring-avoidance
  similarity: 0.65
solutions:
- incremental-refactoring
- technical-debt-backlog
- raising-user-awareness
- security-culture
- security-policies-for-users
- preparatory-refactoring
- team-retrospectives
- workaround-registry
- defect-triage-process
- debt-accrual-analysis
- debt-classification
- technical-debt-assessment
layout: problem
lang: de
en_slug: workaround-culture
---

## Description

Workaround-Kultur entwickelt sich, wenn Teams konsequent wählen, temporäre Lösungen zu implementieren oder Probleme zu umgehen, statt ihre Grundursachen anzugehen. Dies schafft ein Umfeld, in dem sich Schichten von Patches, Hacks und Workarounds über die Zeit anhäufen, was das System zunehmend komplex und unvorhersehbar macht. Während einzelne Workarounds wie pragmatische kurzfristige Lösungen erscheinen mögen, schaffen sie kollektiv einen Wartungsalbtraum, der zukünftige Entwicklung schwieriger und fehleranfälliger macht.

## Indicators ⟡
- Lösungen beinhalten häufig das „Umgehen" bestehender Systembeschränkungen
- Code-Kommentare enthalten Phrasen wie „temporärer Fix", „Hack" oder „TODO: später ordentlich beheben"
- Bug-Berichte werden als „wird nicht behoben" mit vorgeschlagenen Workarounds geschlossen
- Neue Features erfordern umfangreiche Workarounds zur Integration mit bestehenden Systemen
- Entwickler diskutieren routinemäßig „den richtigen Weg" versus „den Weg, der funktioniert"

## Symptoms ▲

- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Eine Kultur, die Workarounds normalisiert, produziert direkt eine stetig wachsende Sammlung temporärer Fixes, die dauerhaft werden.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Jeder Workaround fügt technische Schulden hinzu, während sich temporäre Lösungen anhäufen, ohne durch ordentliche Implementierungen ersetzt zu werden.
- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Schichten miteinander verbundener Workarounds schaffen ein fragiles System, in dem Änderungen in einem Bereich unerwartete Fehlschläge anderswo verursachen.
- [Zunehmende Brüchigkeit](zunehmende-bruechigkeit.md)
<br/>  Während sich Workarounds über die Zeit anhäufen, wird das System zunehmend fragiler und schwieriger sicher zu ändern.
- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Workaround-belasteter Code ist schwerer zu verstehen, weil die Logik Patches um Probleme widerspiegelt statt sauberes Design.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Die Wartung mehrerer Schichten von Workarounds erfordert erheblich mehr Aufwand als die Wartung ordentlich designter Lösungen.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Wenn Bugs in workaround-lastigem Code auftreten, ist das Nachverfolgen der Grundursache durch Schichten von Patches und Hacks extrem schwierig.

## Causes ▼

- [Termindruck](termindruck.md)
<br/>  Enge Termine drängen Teams, schnelle Workarounds zu implementieren, statt Zeit in ordentliche Lösungen zu investieren.
- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Wenn Teams das Refactoring bestehenden Codes vermeiden, werden Workarounds zum Standardansatz im Umgang mit Designproblemen.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Teams, die Angst haben, bestehende Systeme zu modifizieren, bevorzugen das Hinzufügen von Workarounds obendrauf, statt Grundursachen zu beheben.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Organisatorische Betonung kurzfristiger Lieferung über langfristige Qualität begünstigt Workarounds als Weg des geringsten Widerstands.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Ohne Testabdeckung können Teams Code nicht sicher refaktorieren und greifen zu Workarounds, um Regressionen zu vermeiden.
- [Zeitdruck](zeitdruck.md)
<br/>  Zeitdruck ist ein fundamentaler Treiber der Workaround-Kultur.

## Detection Methods ○
- **Code-Musteranalyse:** Suche nach häufigen Workaround-Indikatoren in Code-Kommentaren und -Struktur
- **Verfolgung technischer Schulden:** Überwachung der Anhäufung temporärer Lösungen, die dauerhaft werden
- **Änderungsauswirkungsanalyse:** Identifikation von Bereichen, wo einfache Änderungen komplexe Workarounds erfordern
- **Entwicklerbefragungen:** Befragung von Teammitgliedern zu ihrer Erfahrung mit Workarounds versus ordentlichen Lösungen
- **Dokumentationsüberprüfung:** Suche nach übermäßiger Komplexität in Setup- oder Deployment-Prozeduren aufgrund von Workarounds

## Examples

Eine Webanwendung muss sich mit einem Legacy-Mainframe-System integrieren, das Daten nur in einem spezifischen Format fester Breite akzeptiert. Statt einen ordentlichen Adapter-Service zu erstellen, fügen Entwickler Formatierungslogik direkt in mehrere Controller in der gesamten Anwendung ein. Über die Zeit wird dieser Workaround erweitert, um Randfälle, Fehlerbedingungen und verschiedene Datentypen zu handhaben, was zu duplizierter Formatierungslogik führt, die über Dutzende von Dateien verstreut ist. Als das Mainframe-System schließlich aktualisiert wird, um JSON zu akzeptieren, entdeckt das Team, dass es Formatierungslogik an 47 verschiedenen Stellen ändern muss. Ein weiteres Beispiel betrifft eine Datenbank, die Performance-Probleme mit bestimmten Abfragemustern hat. Statt die Datenbank zu optimieren oder die zugrunde liegenden Schemadesignprobleme zu beheben, implementieren Entwickler zunehmend komplexe Caching-Schichten, Abfrage-Umschreibungslogik und Hintergrundverarbeitungsjobs, um die Performance-Probleme zu umgehen. Diese Workarounds schaffen ein fragiles System, in dem scheinbar nicht verwandte Änderungen Cache-Invalidierungsprobleme oder Hintergrundjob-Fehlschläge verursachen können, was das System weit schwieriger zu warten macht, als wenn die ursprünglichen Datenbankprobleme ordentlich angegangen worden wären.
