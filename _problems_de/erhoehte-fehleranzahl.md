---
title: Erhöhte Fehleranzahl
description: Änderungen führen häufiger neue Defekte ein, was zu einer höheren Fehlerrate
  in Produktion und verschlechterter Softwarequalität führt.
category:
- Code
- Process
related_problems:
- slug: high-bug-introduction-rate
  similarity: 0.8
- slug: increased-error-rates
  similarity: 0.7
- slug: slow-development-velocity
  similarity: 0.7
- slug: increased-risk-of-bugs
  similarity: 0.7
- slug: increased-cost-of-development
  similarity: 0.65
- slug: high-defect-rate-in-production
  similarity: 0.65
solutions:
- contract-testing
- development-workflow-automation
- regression-testing
- code-hotspot-analysis
- small-change-batches
- code-reviews
- code-quality-gates
- characterization-tests
- change-impact-analysis
- production-like-test-data
- defect-triage-process
- duplication-detection
- typed-schema-extraction
layout: problem
lang: de
en_slug: increased-bug-count
---

## Description

Erhöhte Fehleranzahl tritt auf, wenn Softwareänderungen durchgängig neue Defekte in einer höheren Rate einführen, als normale Entwicklungsprozesse produzieren sollten. Dieses Problem äußert sich als wachsende Anzahl gemeldeter Probleme, häufige Produktionsvorfälle und einen allgemeinen Rückgang der Softwarequalität. Die erhöhte Fehlerrate deutet oft auf zugrunde liegende Probleme mit Entwicklungspraktiken, Codequalität oder Systemarchitektur hin, die die Software fehleranfälliger machen.

## Indicators ⟡

- Fehlerberichte nehmen über die Zeit zu, trotz ähnlicher Entwicklungsaktivitätsniveaus
- Neue Features führen durchgängig unerwartete Nebeneffekte ein
- Produktionsvorfälle treten nach Releases häufiger auf
- Tests entdecken mehr Defekte als historisch normal
- Fehlerbehebungszyklen werden länger und komplexer

## Symptoms ▲

- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Mehr Fehler bedeuten mehr Zeit und Geld, die für Debugging und Behebung statt für den Bau neuer Features aufgewendet werden.
- [Erhöhte Last im Kundensupport](erhoehte-last-im-kundensupport.md)
<br/>  Mehr Produktionsfehler führen dazu, dass mehr Nutzer mit Problemen den Support kontaktieren.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Eine höhere Fehlerrate verschlechtert das Nutzererlebnis, was zu Nutzerfrustration und Beschwerden führt.
- [Release-Instabilität](release-instabilitaet.md)
<br/>  Häufige neue Defekte machen jedes Release weniger stabil und wahrscheinlicher, Produktionsprobleme zu verursachen.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Wenn Änderungen durchgängig Fehler einführen, werden Entwickler zurückhaltend, die Codebasis zu ändern.
- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Mehr während der Entwicklung eingeführte Fehler übersetzen sich direkt in mehr in der Live-Umgebung entdeckte Defekte.

## Causes ▼

- [Zunehmende Brüchigkeit](zunehmende-bruechigkeit.md)
<br/>  Eine brüchige Codebasis bedeutet, dass kleine Änderungen unvorhersehbare weitreichende Auswirkungen haben, was mehr Fehler einführt.
- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Unzureichende Testabdeckung erlaubt es mehr Fehlern, unentdeckt in die Produktion zu gelangen.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Eng gekoppelte Komponenten bedeuten, dass Änderungen in einem Bereich unbeabsichtigt andere brechen, was die Fehlereinführung vervielfacht.
- [Erhöhte kognitive Last](erhoehte-kognitive-last.md)
<br/>  Wenn Entwickler Schwierigkeiten haben, den Code zu verstehen, machen sie mit höherer Wahrscheinlichkeit Fehler, die Fehler einführen.
- [Unzureichendes Code-Review](unzureichendes-code-review.md)
<br/>  Unzureichendes Code-Review erlaubt es Fehlern, unentdeckt durch den Entwicklungsprozess zu gelangen, was direkt zur erhöhten Fehleranzahl beiträgt.

## Detection Methods ○

- **Fehlerverfolgungsanalyse:** Beobachtung von Fehlerberichtstrends, Schweregradverteilungen und Zeit-bis-zur-Lösung-Metriken
- **Release-Qualitätsmetriken:** Nachverfolgung von Defekten pro Release und Fehlerdichte in unterschiedlichen Codebereichen
- **Produktionsvorfall-Tracking:** Überwachung der Häufigkeit und Schwere von Produktionsproblemen
- **Kundensupport-Metriken:** Analyse von Support-Ticket-Volumen und Arten gemeldeter Probleme
- **Codequalitätsmetriken:** Nutzung statischer Analysewerkzeuge zur Identifikation potenziell problematischer Codebereiche

## Examples

Eine E-Commerce-Plattform, die zuvor durchschnittlich 5 Fehlerberichte pro Release hatte, hat nun durchgängig 20+ gemeldete Fehler innerhalb der ersten Woche jedes Deployments. Die Untersuchung zeigt, dass schnelle Feature-Entwicklung komplexe gegenseitige Abhängigkeiten zwischen Warenkorb-, Bestands- und Zahlungssystemen eingeführt hat, was dazu führt, dass scheinbar unzusammenhängende Änderungen Funktionalität auf unerwartete Weise brechen. Ein weiteres Beispiel betrifft ein Content-Management-System, bei dem kürzliche Performance-Optimierungen subtile Datenverfälschungsprobleme eingeführt haben, die nur unter bestimmten Lastbedingungen auftreten, was zu einem 300%igen Anstieg kundenseitig gemeldeter Dateninkonsistenzen führt.
