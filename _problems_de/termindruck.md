---
title: Termindruck
description: Intensiver Druck, Termine einzuhalten, führt zu überstürzten Entscheidungen,
  Abkürzungen und Qualitätseinbußen in der Softwareentwicklung.
category:
- Code
- Management
- Process
related_problems:
- slug: time-pressure
  similarity: 0.85
- slug: unrealistic-deadlines
  similarity: 0.75
- slug: increased-technical-shortcuts
  similarity: 0.7
- slug: market-pressure
  similarity: 0.7
- slug: lower-code-quality
  similarity: 0.65
- slug: high-technical-debt
  similarity: 0.65
solutions:
- formal-change-control-process
- iterative-development
- short-iteration-cycles
- capacity-based-planning
- explicit-prioritization-framework
- improvement-budget
- regular-stakeholder-demonstrations
- work-in-progress-limits
layout: problem
lang: de
en_slug: deadline-pressure
---

## Description

Termindruck entsteht, wenn Entwicklungsteams intensiven Zeiteinschränkungen ausgesetzt sind, die sie zwingen, Geschwindigkeit über Qualität zu stellen, was zu überstürzten Implementierungen, übersprungenen Best Practices und der Anhäufung technischer Schulden führt. Während etwas Termindruck Teams motivieren kann, führt übermäßiger Druck durchgängig zu schlechter Entscheidungsfindung, erhöhtem Stress und langfristigen Problemen, die die Entwicklung letztlich mehr verlangsamen, als die ursprüngliche Zeitersparnis eingebracht hat.

## Indicators ⟡

- Das Team arbeitet durchgängig Überstunden, um Termine einzuhalten
- Code-Reviews werden verkürzt oder übersprungen, um Zeit zu sparen
- Testphasen werden komprimiert oder eliminiert
- Design- und Planungsaktivitäten werden überstürzt oder umgangen
- Das Team äußert Angst, unrealistische Zeitpläne einzuhalten

## Symptoms ▲

- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Das Hetzen, um Termine einzuhalten, veranlasst Teams, Abkürzungen zu nehmen und ordentliche Implementierungen aufzuschieben, was technische Schulden anhäuft.
- [Qualitätskompromisse](qualitaetskompromisse.md)
<br/>  Unter Termindruck senken Teams absichtlich Qualitätsstandards, indem sie Tests, Reviews und ordentliches Design überspringen.
- [Erhöhter Stress und Burnout](erhoehter-stress-und-burnout.md)
<br/>  Anhaltender Termindruck führt dazu, dass Teammitglieder Überstunden machen und chronischen Stress erleben, was zu Burnout führt.
- [Zunehmende technische Abkürzungen](zunehmende-technische-abkuerzungen.md)
<br/>  Teams setzen Schnelllösungen und Workarounds statt ordentlicher Lösungen um, um enge Termine einzuhalten.
- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Gehetzte Entwickler überspringen Code-Reviews, Tests und ordentliches Design, was zu mehr Defekten und schwerer wartbarem Code führt.
- [Hohe Rate an neu eingeführten Fehlern](hohe-rate-an-neu-eingefuehrten-fehlern.md)
<br/>  Das Hetzen unter Druck führt dazu, dass Entwickler mehr Fehler machen und Validierungsschritte überspringen, was mehr Fehler einführt.
- [Verzögerte Fehlerbehebungen](verzoegerte-fehlerbehebungen.md)
<br/>  Termindruck führt dazu, dass Teams die Feature-Lieferung über Fehlerbehebungen priorisieren, was direkt zu verzögerten Fehlerbehebungen führt.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Unter wiederholtem Zeitdruck wählen Entwickler wiederholt schnelle Workarounds statt ordentlicher Fixes, und diese individuellen Entscheidungen summieren sich zu einem wachsenden Geflecht aus Workarounds.
- [Probleme mit algorithmischer Komplexität](probleme-mit-algorithmischer-komplexitaet.md)
<br/>  Zeitdruck führt dazu, dass Entwickler die erste funktionierende Lösung umsetzen, ohne deren algorithmische Effizienz zu berücksichtigen.
- [Annahmenbasierte Entwicklung](annahmenbasierte-entwicklung.md)
<br/>  Wenn ein knapper Zeitplan gerade in die Phase der Anforderungsklärung fällt, setzen Entwickler auf unvalidierten Annahmen fort, statt sich Zeit zu nehmen, das Verständnis mit Stakeholdern abzustimmen.

## Causes ▼

- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Eine Management-Kultur, die durchgängig sofortige Feature-Lieferung über Nachhaltigkeit stellt, setzt und hält aggressive Liefertermine als primären Hebel zur Durchsetzung dieser Priorität aufrecht.
- [Unrealistische Termine](unrealistische-termine.md)
<br/>  Das Management setzt Termine, die den tatsächlich benötigten Entwicklungsaufwand nicht berücksichtigen, was intensiven Druck auf das Team erzeugt.
- [Schlechte Planung](schlechte-planung.md)
<br/>  Fehlende realistische Arbeitsschätzungen und Projektplanung führen zu Zeitplänen, die für den tatsächlichen Umfang zu komprimiert sind.
- [Scope Creep](scope-creep.md)
<br/>  Die Ausweitung von Anforderungen ohne Anpassung der Zeitpläne erzeugt zunehmenden Druck, da mehr Arbeit bis zum selben Termin abgeschlossen werden muss.
- [Marktdruck](marktdruck.md)
<br/>  Externe Wettbewerbskräfte treiben Organisationen dazu, aggressive Termine zu setzen, um Wettbewerbern am Markt zuvorzukommen.

## Detection Methods ○

- **Überstunden-Tracking:** Beobachtung der Arbeitszeiten und Stressindikatoren des Teams
- **Korrelation von Qualitätsmetriken:** Vergleich von Codequalitätsmetriken mit Terminperioden
- **Anhäufung technischer Schulden:** Nachverfolgung, wann technische Schulden im Verhältnis zum Termindruck zunehmen
- **Team-Stress-Umfragen:** Regelmäßige Bewertung von Teamstressniveaus und Terminsorgen
- **Entscheidungsqualitätsanalyse:** Bewertung der Qualität technischer Entscheidungen, die unter Zeitdruck getroffen werden

## Examples

Einem Entwicklungsteam werden vier Wochen gegeben, um ein komplexes Zahlungsabwicklungs-Feature umzusetzen, das normalerweise acht Wochen dauern würde, um es ordentlich zu machen. Unter intensivem Termindruck überspringen sie das Schreiben von Unit-Tests, setzen schnelle und unsaubere Fehlerbehandlung um und nutzen ein einfaches, aber ineffizientes Datenbankdesign. Das Feature wird pünktlich ausgeliefert, verursacht aber sofort Performance-Probleme in der Produktion. Die Behebung der Performance-Probleme erfordert drei zusätzliche Wochen Arbeit und führt Fehler ein, weil der Code nicht ordentlich getestet wurde. Die "Zeitersparnis" durch das Hetzen kostete tatsächlich langfristig mehr Zeit und schädigte die Glaubwürdigkeit des Teams. Ein weiteres Beispiel betrifft ein Team, das mit einem kritischen Geschäftstermin konfrontiert ist und beschließt, ein bestehendes Codemodul zu kopieren und zu modifizieren, statt eine ordentliche Abstraktion zu entwerfen. Der kopierte Code funktioniert für den unmittelbaren Bedarf, schafft aber Wartungsaufwand, weil beide Module für künftige Änderungen aktualisiert werden müssen. Sechs Monate später hat das Team mehr Zeit mit der Wartung des duplizierten Codes verbracht, als es gekostet hätte, ursprünglich eine ordentliche Lösung umzusetzen.
