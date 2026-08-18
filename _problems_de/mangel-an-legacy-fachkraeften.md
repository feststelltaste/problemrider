---
title: Mangel an Legacy-Fachkräften
description: Kritischer Mangel an Entwicklern mit Wissen über Legacy-Technologien
  schafft Engpässe und Single Points of Failure für die Systemwartung.
category:
- Management
- Team
related_problems:
- slug: skill-development-gaps
  similarity: 0.7
- slug: technology-isolation
  similarity: 0.65
- slug: vendor-dependency-entrapment
  similarity: 0.65
- slug: legacy-system-documentation-archaeology
  similarity: 0.65
- slug: obsolete-technologies
  similarity: 0.65
- slug: technology-stack-fragmentation
  similarity: 0.65
solutions:
- architecture-roadmap
- cross-functional-skill-development
- technical-skills-development
- emulation
- platform-independent-programming-languages
- security-training
- standard-software
- knowledge-rotation
- communities-of-practice
- internal-technical-coaching
- code-reading-sessions
- risk-quantification
- cost-of-delay
- executive-sponsorship
- modernization-options-comparison
- continuous-dependency-updates
layout: problem
lang: de
en_slug: legacy-skill-shortage
---

## Description

Mangel an Legacy-Fachkräften tritt auf, wenn Organisationen mit einer kritischen Knappheit an Entwicklern und technischem Personal konfrontiert sind, die veraltete Programmiersprachen, Plattformen und Technologien verstehen, von denen ihre Legacy-Systeme abhängen. Dieses Problem schafft schwerwiegendes operatives Risiko, während die verbleibenden erfahrenen Fachkräfte in Rente gehen, den Beruf wechseln oder nicht verfügbar werden, was Organisationen unfähig macht, kritische Systeme zu warten, zu modifizieren oder Probleme zu beheben. Anders als allgemeine Wissenslücken betrifft dies Fähigkeiten, die nicht mehr an Schulen gelehrt werden und auf dem Arbeitsmarkt zunehmend selten sind.

## Indicators ⟡

- Schwierigkeiten, Auftragnehmer oder Mitarbeiter mit Erfahrung in den Legacy-Technologien der Organisation zu finden
- Legacy-Systemwartungsarbeit, konzentriert auf wenige Senior-Mitarbeiter nahe der Rente
- Steigende Auftragnehmerraten für Legacy-Technologie-Spezialisten
- Stellenausschreibungen für Legacy-Fähigkeiten, die monatelang unbesetzt bleiben
- Schulungsprogramme für Legacy-Technologien, die nicht mehr existieren oder unerschwinglich teuer sind
- Informatik-Studiengänge an Universitäten, die die benötigten Legacy-Sprachen oder -Plattformen nicht mehr lehren
- Support für Legacy-Technologie-Anbieter, der eingestellt wird oder bereits geendet hat

## Symptoms ▲

- [Wartungsengpässe](wartungsengpaesse.md)
<br/>  Mit nur wenigen Personen, die Legacy-Systeme warten können, läuft alle Wartungsarbeit durch sie, was schwerwiegende Engpässe schafft.
- [Single Points of Failure](single-points-of-failure.md)
<br/>  Wenn nur ein oder zwei Personen ein Legacy-System verstehen, blockiert ihre Nichtverfügbarkeit allen Fortschritt an diesem System.
- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Knappe Legacy-Fähigkeiten verlangen Premium-Sätze, und die wenigen verfügbaren Spezialisten brauchen länger, aufgrund fehlender Peer-Unterstützung.
- [Langsame Vorfallslösung](langsame-vorfallsloesung.md)
<br/>  Wenn Vorfälle in Legacy-Systemen auftreten, verzögert sich die Lösung, weil wenige Personen die Expertise haben, Probleme zu diagnostizieren.
- [Lähmung der Modernisierungsstrategie](laehmung-der-modernisierungsstrategie.md)
<br/>  Ohne genug qualifizierte Personen, um sowohl das Legacy-System zu warten als auch einen Ersatz zu bauen, können sich Organisationen nicht auf einen Modernisierungspfad festlegen.

## Causes ▼

- [Veraltete Technologien](veraltete-technologien.md)
<br/>  Systeme, die auf veralteten Technologien aufgebaut sind, die nicht mehr gelehrt oder weit genutzt werden, schaffen einen schrumpfenden Pool qualifizierter Entwickler.
- [Technologie-Isolation](technologie-isolation.md)
<br/>  Systeme, die von modernen Technologie-Stacks isoliert sind, sind für Entwickler unattraktiv, was die Rekrutierung neuer Talente erschwert.
- [Hohe Fluktuation](hohe-fluktuation.md)
<br/>  Erfahrene Legacy-Entwickler, die die Organisation verlassen, beschleunigen den Fähigkeitsmangel, während institutionelles Wissen mit ihnen geht.
- [Wissenssilos](wissenssilos.md)
<br/>  Wissen, das bei wenigen Entwicklern konzentriert ist, die gehen, nimmt unersetzliche Legacy-Expertise mit sich, was direkt zum Problem beiträgt.

## Detection Methods ○

- Durchführung von Fähigkeitsinventar-Bewertungen für alle kritischen Legacy-Technologien in der Organisation
- Überwachung der Altersdemografie von Mitarbeitern mit Legacy-Systemexpertise und Rentenzeitplänen
- Nachverfolgung von Rekrutierungsschwierigkeiten und Besetzungszeit für Legacy-Technologie-Positionen
- Bewertung der Marktverfügbarkeit und Kostentrends für Legacy-Technologie-Auftragnehmer und -Berater
- Befragung aktueller Legacy-qualifizierter Mitarbeiter zu Nachfolgeplanung und Wissenstransfer-Bedürfnissen
- Bewertung der Schulungsverfügbarkeit und -kosten, um neue Mitarbeiter mit Legacy-Technologien vertraut zu machen
- Überwachung der Anbieter-Support-Lebenszyklen für Legacy-Plattformen und -Technologien
- Bewertung der Geschäftsrisikoexposition durch Verlust von Legacy-Schlüsselexpertise

## Examples

Das Steuerverarbeitungssystem einer Regierungsbehörde läuft auf einem Mainframe mit COBOL-Code aus den 1970er-Jahren. Die drei Entwickler, die das System verstehen, sind 67, 64 und 58 Jahre alt, wobei der Senior-Entwickler plant, in 18 Monaten in Rente zu gehen. Als sie Stellenausschreibungen für COBOL-Programmierer veröffentlichen, erhalten sie keine qualifizierten Bewerber, trotz überdurchschnittlicher Gehälter. Die örtliche Universität hat COBOL seit 15 Jahren nicht mehr gelehrt, und die wenigen Auftragnehmer mit COBOL-Erfahrung verlangen 200+ $ pro Stunde und haben monatelange Wartelisten. Während der Steuersaison schlägt ein kritischer Batch-Verarbeitungsjob fehl, und das Team verbringt 72 Stunden mit Fehlerbehebung, weil der Fehler in einem Codeabschnitt auftritt, mit dem selbst der Senior-Entwickler seit einem Jahrzehnt nicht gearbeitet hat. Die Behörde erkennt, dass sie 18 Monate hat, um entweder Ersatzexpertise zu finden, neue Mitarbeiter in veralteten Technologien zu schulen, oder eine Systemmodernisierung abzuschließen, die ursprünglich 5 Jahre dauern sollte. Das Risiko, die Steuerverarbeitungsfähigkeit während der Hochsaison zu verlieren, schafft eine Krise, die Notfall-Budgetzuweisung sowohl für Fähigkeitserwerb als auch beschleunigte Modernisierungsbemühungen erzwingt.
