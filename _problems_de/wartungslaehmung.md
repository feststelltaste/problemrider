---
title: Wartungslähmung
description: Teams vermeiden notwendige Verbesserungen, weil sie nicht verifizieren
  können, dass Änderungen bestehende Funktionalität nicht brechen.
category:
- Code
- Process
related_problems:
- slug: resistance-to-change
  similarity: 0.75
- slug: fear-of-change
  similarity: 0.75
- slug: refactoring-avoidance
  similarity: 0.75
- slug: fear-of-breaking-changes
  similarity: 0.7
- slug: decision-paralysis
  similarity: 0.7
- slug: inability-to-innovate
  similarity: 0.7
solutions:
- architecture-roadmap
- regression-testing
- characterization-tests
- improvement-budget
- mikado-method
- preparatory-refactoring
- code-hotspot-analysis
- dependency-breaking-techniques
- parallel-run
- pilot-projects
- technical-debt-assessment
- debt-classification
- debt-remediation-estimation
- automated-code-migration
- large-scale-refactoring
layout: problem
lang: de
en_slug: maintenance-paralysis
---

## Description

Wartungslähmung tritt auf, wenn Entwicklungsteams nicht mehr in der Lage sind, notwendige Wartung, Verbesserungen oder Refactoring an ihrer Codebasis durchzuführen, weil ihnen das Vertrauen fehlt, Änderungen sicher vornehmen zu können. Dies schafft eine sich selbst verstärkende Abwärtsspirale, in der die Codebasis zunehmend schwieriger zu warten wird, was zu noch größerem Zögern führt, Änderungen vorzunehmen. Teams finden sich gefangen zwischen dem Bedürfnis, das System zu verbessern, und der Unfähigkeit, dies zu tun, ohne katastrophale Fehlschläge zu riskieren.

## Indicators ⟡
- Entwickler äußern Zurückhaltung, funktionierenden Code zu refaktorieren oder zu verbessern
- Wartungsaufgaben werden wiederholt aufgeschoben oder vermieden
- Das Team diskutiert benötigte Verbesserungen, setzt sie aber nie um
- Fehlerbehebungen werden als minimale Patches statt ordentlicher Lösungen angewendet
- Technische Schulden häufen sich an, während Verbesserungsbemühungen stagnieren

## Symptoms ▲

- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Wenn Teams keine notwendige Wartung und kein Refactoring durchführen können, häufen sich technische Schulden ungebremst an.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Unfähig, Grundprobleme ordentlich zu beheben, implementieren Teams Workarounds, die Komplexität hinzufügen statt Probleme direkt anzugehen.
- [Zunehmende Brüchigkeit](zunehmende-bruechigkeit.md)
<br/>  Die Vermeidung notwendiger Wartung lässt die Codebasis zunehmend fragiler und fehleranfälliger werden.
- [Systemstagnation](systemstagnation.md)
<br/>  Die Angst, Änderungen vorzunehmen, führt zur Stagnation des Systems, das bei Sicherheitspatches, Abhängigkeits-Updates und Verbesserungen zurückbleibt.
- [Stagnierende Architektur](stagnierende-architektur.md)
<br/>  Wenn Teams von Änderungen gelähmt sind, kann sich die Architektur nicht weiterentwickeln, um neue Anforderungen zu erfüllen.

## Causes ▼

- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Ohne automatisierte Tests können Teams nicht verifizieren, dass Änderungen bestehende Funktionalität nicht brechen, was die Lähmung erzeugt.
- [Schlechte Testabdeckung](schlechte-testabdeckung.md)
<br/>  Unzureichende Testabdeckung bedeutet, dass Änderungen nicht validiert werden können, was Teams davor scheuen lässt, das System zu modifizieren.
- [Angst vor Breaking Changes](angst-vor-breaking-changes.md)
<br/>  Vergangene Erfahrungen mit Änderungen, die Fehlschläge verursachten, schaffen eine Angstkultur, die notwendige Wartung verhindert.
- [Wissenssilos](wissenssilos.md)
<br/>  Wenn nur ausgeschiedene Entwickler das System verstanden, fehlt aktuellen Teams das Vertrauen, sichere Änderungen vorzunehmen.
- [Geschichte fehlgeschlagener Änderungen](geschichte-fehlgeschlagener-aenderungen.md)
<br/>  Eine Historie von Änderungen, die Produktionsfehlschläge verursachten, verstärkt den Glauben, dass es sicherer sei, nichts zu ändern.

## Detection Methods ○
- **Änderungshäufigkeitsanalyse:** Messung, wie oft Wartungsaufgaben vorgeschlagen versus abgeschlossen werden
- **Nachverfolgung technischer Schulden:** Überwachung der Anhäufung bekannter Probleme, die unbehandelt bleiben
- **Entwicklerbefragungen:** Befragung von Teammitgliedern zu ihrem Wohlfühlgrad bei Systemänderungen
- **Code-Alters-Analyse:** Identifikation kritischen Codes, der trotz bekannter Probleme nicht aktualisiert wurde
- **Risikobewertungs-Reviews:** Nachverfolgung von Diskussionen über benötigte Änderungen, die als „zu riskant" eingestuft werden

## Examples

Ein Finanzdienstleistungsunternehmen hat ein kritisches Transaktionsverarbeitungssystem, das vor 8 Jahren von Entwicklern geschrieben wurde, die das Unternehmen seitdem verlassen haben. Das System verarbeitet täglich Millionen von Dollar, hat aber keine automatisierten Tests und nutzt veraltete Bibliotheken mit bekannten Sicherheitslücken. Das aktuelle Team weiß, dass die Bibliotheken aktualisiert werden müssen und mehrere Performance-Verbesserungen vorgenommen werden könnten, ist aber gelähmt von der Angst, dass jede Änderung Transaktionsfehler oder Datenkorruption verursachen könnte. Sie fahren fort, minimale Fehlerbehebungen anzuwenden, während das System zunehmend brüchiger wird und die technischen Schulden wachsen. In einem anderen Beispiel hat eine Gesundheitsanwendung Code zur Patientendatenverwaltung, bei dem alle übereinstimmen, dass er für bessere Wartbarkeit refaktoriert werden muss, aber das Fehlen umfassender Tests und die lebenswichtige Natur der Daten machen das Team unwillig, den funktionierenden Code anzufassen, obwohl es zunehmend schwieriger wird, neue Features hinzuzufügen oder Fehler zu beheben.
