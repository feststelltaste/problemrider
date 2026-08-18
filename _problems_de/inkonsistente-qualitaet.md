---
title: Inkonsistente Qualität
description: Manche Teile des Systems sind gut gewartet, während andere sich verschlechtern,
  was zu unvorhersehbaren Nutzererlebnissen und Wartungsherausforderungen führt.
category:
- Code
- Process
related_problems:
- slug: inconsistent-coding-standards
  similarity: 0.7
- slug: inconsistent-execution
  similarity: 0.7
- slug: quality-degradation
  similarity: 0.7
- slug: inconsistent-behavior
  similarity: 0.7
- slug: inconsistent-codebase
  similarity: 0.65
- slug: inconsistent-knowledge-acquisition
  similarity: 0.65
solutions:
- definition-of-done
- checklists
- portability-checklists
- secure-software-development
- security-certification
- security-frameworks
- security-policies-for-development
- code-quality-gates
layout: problem
lang: de
en_slug: inconsistent-quality
---

## Description

Inkonsistente Qualität tritt auf, wenn unterschiedliche Teile eines Softwaresystems dramatisch unterschiedliche Qualitäts-, Wartungs- und Zuverlässigkeitsniveaus aufweisen. Dies schafft einen Flickenteppich-Effekt, bei dem manche Komponenten robust und gut gestaltet sind, während andere brüchig, schlecht dokumentiert oder schwer zu warten sind. Diese Inkonsistenz entsteht oft, wenn es keinen systematischen Ansatz für Qualitätsstandards gibt oder wenn unterschiedliche Teams oder Personen unterschiedliche Sorgfaltsgrade bei ihrer Arbeit walten lassen.

## Indicators ⟡

- Manche Systemmodule sind zuverlässig, während andere häufig kaputtgehen
- Die Codequalität variiert dramatisch zwischen unterschiedlichen Teilen der Codebasis
- Das Nutzererlebnis unterscheidet sich erheblich über unterschiedliche Features hinweg
- Manche Bereiche haben umfassende Tests, während andere keine haben
- Die Dokumentationsqualität variiert stark über unterschiedliche Komponenten hinweg

## Symptoms ▲

- [Nutzerverwirrung](nutzerverwirrung.md)
<br/>  Nutzer stoßen auf unterschiedliche Qualitätsniveaus über Features hinweg, was zu unvorhersehbaren Erfahrungen und Vertrauensverlust führt.
- [Erhöhte Fehleranzahl](erhoehte-fehleranzahl.md)
<br/>  Die minderwertigen Teile des Systems produzieren mehr Defekte, was die Gesamtfehleranzahl erhöht.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Entwickler bekommen Angst, brüchige, minderwertige Abschnitte der Codebasis zu ändern, wegen des hohen Risikos, etwas zu brechen.
- [Erhöhte Last im Kundensupport](erhoehte-last-im-kundensupport.md)
<br/>  Nutzer, die auf Probleme in den minderwertigeren Teilen des Systems stoßen, erzeugen mehr Support-Anfragen.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Die unzuverlässigen Teile des Systems frustrieren Nutzer und schädigen ihre Gesamtwahrnehmung des Produkts.

## Causes ▼

- [Fehlende Eigenverantwortung und Rechenschaftspflicht](fehlende-eigenverantwortung-und-rechenschaftspflicht.md)
<br/>  Ohne klare Eigenverantwortung erhalten manche Systembereiche Aufmerksamkeit, während andere vernachlässigt werden und sich verschlechtern.
- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Variierende Coding-Standards über die Codebasis hinweg führen zu unterschiedlichen Codequalitätsniveaus in unterschiedlichen Modulen.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Angehäufte technische Schulden konzentrieren sich in bestimmten Bereichen, was dazu führt, dass diese Teile sich verschlechtern, während gewartete Bereiche gesund bleiben.
- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Ungleiche Testabdeckung bedeutet, dass manche Teile des Systems umfassende Qualitätssicherung haben, während andere keine haben.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Teams mit unerfahrenen Entwicklern produzieren inkonsistente Qualität, weil Fähigkeitsniveaus variieren und weniger erfahrene Entwickler mehr Probleme einführen.

## Detection Methods ○

- **Qualitätsmetrik-Analyse:** Vergleich von Codequalitätsmetriken (Komplexität, Testabdeckung, Fehlerraten) über unterschiedliche Systemkomponenten hinweg
- **Nutzerfeedback-Analyse:** Nachverfolgung von Nutzerbeschwerden und Zufriedenheitswerten für unterschiedliche Features
- **Entwicklerbefragungen:** Befragung von Teammitgliedern zu ihrer Erfahrung bei der Arbeit mit unterschiedlichen Teilen des Systems
- **Code-Review-Muster:** Analyse der Arten und Häufigkeit von Problemen, die in Reviews für unterschiedliche Bereiche gefunden werden
- **Wartungsaufwands-Tracking:** Überwachung, wie viel Zeit für die Wartung unterschiedlicher Systemkomponenten aufgewendet wird

## Examples

Eine Finanzanwendung hat ein modernes, gut getestetes Zahlungsverarbeitungsmodul mit umfassender Fehlerbehandlung und Logging, während das Kontoverwaltungssystem eine schlecht dokumentierte Legacy-Komponente mit minimalen Tests und häufigen Fehlern ist. Nutzer erleben reibungslose Zahlungsabläufe, stoßen aber ständig auf Probleme beim Aktualisieren ihrer Profilinformationen. Ein weiteres Beispiel betrifft eine E-Commerce-Plattform, bei der die Produktkatalogsuche schnell und zuverlässig ist, der Warenkorb aber häufig Artikel verliert und verwirrendes Verhalten hat, was zu Kundenbeschwerden und abgebrochenen Käufen führt.
