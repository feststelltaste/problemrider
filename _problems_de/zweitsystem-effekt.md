---
title: Zweitsystem-Effekt
description: Lehren aus einem alten System führen zu Überkompensation und schaffen
  aufgeblähte oder übermäßig ambitionierte Designs.
category:
- Architecture
- Code
- Process
related_problems:
- slug: complex-implementation-paths
  similarity: 0.6
- slug: cognitive-overload
  similarity: 0.6
- slug: feature-creep
  similarity: 0.6
- slug: accumulation-of-workarounds
  similarity: 0.6
- slug: feature-bloat
  similarity: 0.6
- slug: ripple-effect-of-changes
  similarity: 0.6
solutions:
- architecture-reviews
- architecture-roadmap
- boring-technologies
- mikado-method
- parallel-run
- technology-radar
- lightweight-design-review
- pilot-projects
- modernization-options-comparison
- no-regret-moves
- staged-investment-with-decision-gates
layout: problem
lang: de
en_slug: second-system-effect
---

## Description

Der Zweitsystem-Effekt tritt auf, wenn Architekten und Entwickler, die aus den Beschränkungen und Problemen eines Vorgängersystems gelernt haben, überkompensieren, indem sie einen übermäßig komplexen, funktionsreichen Ersatz designen, der versucht, jedes denkbare Problem zu lösen. Dies resultiert oft in Systemen, die schwerer zu bauen, zu warten und zu verstehen sind als nötig. Der Effekt ist besonders häufig bei Modernisierungsprojekten von Legacy-Systemen, bei denen Teams versuchen, alle vergangenen Schmerzpunkte gleichzeitig anzugehen, statt inkrementell zu bauen.

## Indicators ⟡

- Design-Dokumente, die erheblich komplexer sind, als die Geschäftsanforderungen rechtfertigen
- Anforderungen, die die Lösung von Problemen beinhalten, die aktuell nicht existieren oder hypothetisch sind
- Architektur-Meetings, die häufig auf „gelernte Lektionen" aus dem alten System verweisen
- Feature-Listen, die während der Planungsphasen exponentiell wachsen
- Stakeholder, die Bedenken äußern, dass das neue System „over-engineered" wirkt
- Entwicklungsschätzungen, die 3-5x größer als erwartet für scheinbar einfache Ersetzungen sind

## Symptoms ▲

- [Verzögerte Projektzeitpläne](verzoegerte-projektzeitplaene.md)
<br/>  Überambitionierte Designs für das Ersatzsystem brauchen viel länger zur Implementierung als geplant, was Zeitpläne weit über Schätzungen hinaus schiebt.
- [Feature-Aufblähung](feature-aufblaehung.md)
<br/>  Das Ersatzsystem wird mit Features aufgebläht, die hypothetische Probleme aus dem alten System statt tatsächliche Geschäftsbedürfnisse adressieren.
- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Over-engineerte Ersatzsysteme erfordern erheblich mehr Entwicklungsressourcen als nötig, um Kernfunktionalität zu liefern.
- [Wartungs-Overhead](wartungs-overhead.md)
<br/>  Komplexe, überdesignte Ersatzsysteme schaffen laufende Wartungslast für Features und Abstraktionen, die selten genutzt werden.
- [Verschwendeter Entwicklungsaufwand](verschwendeter-entwicklungsaufwand.md)
<br/>  Erheblicher Aufwand wird in den Bau fortgeschrittener Fähigkeiten investiert, die Nutzer nie tatsächlich verwenden, was reine Verschwendung darstellt.

## Causes ▼

- [Negative Erfahrungen aus der Vergangenheit](negative-erfahrungen-aus-der-vergangenheit.md)
<br/>  Schmerzhafte Erfahrungen mit den Beschränkungen des ursprünglichen Systems treiben Teams dazu, zu überkompensieren, indem sie versuchen, jedes mögliche Problem im Ersatz zu verhindern.
- [Gold Plating](gold-plating.md)
<br/>  Entwickler fügen dem neuen System unnötige Features und Komplexität hinzu, weil sie jedes denkbare Problem lösen wollen, dem sie im alten begegnet sind.
- [Annahmenbasierte Entwicklung](annahmenbasierte-entwicklung.md)
<br/>  Teams treffen Annahmen darüber, was das neue System braucht, basierend auf Schmerzpunkten des alten Systems, statt tatsächliche aktuelle Anforderungen zu validieren.

## Detection Methods ○

- Regelmäßige Überprüfung von Feature-zu-Geschäftswert-Verhältnissen während der Planung
- Vergleich von Komplexitätsmetriken zwischen alten und neuen Systemdesigns
- Durchführung von Architektur-Reviews mit externen Experten, die mit dem Legacy-System nicht vertraut sind
- Verfolgung der Entwicklungsgeschwindigkeit im Vergleich zu einfacheren Alternativansätzen
- Überwachung des Stakeholder-Feedbacks zu Systemkomplexität und Nutzbarkeit
- Nutzung von Prototyping zur Validierung, ob komplexe Features tatsächlich benötigt werden
- Messung der Time-to-Market für Basisfunktionalität im Vergleich zu Wettbewerbern

## Examples

Ein Einzelhandelsunternehmen, das sein Legacy-Bestandsverwaltungssystem ersetzt, entscheidet sich, eine neue Plattform zu bauen, die nicht nur Bestand handhabt, sondern auch prädiktive Analytik, KI-gestützte Nachfrageprognose, Blockchain-basierte Lieferkettenverfolgung und eine flexible Regel-Engine für beliebige zukünftige Geschäftslogikänderungen umfasst. Während das alte System Beschränkungen bei Berichterstattung und Integration hatte, wird das neue System so komplex, dass es drei Jahre statt der geplanten 18 Monate dauert, es zu bauen. Bei der finalen Bereitstellung nutzen die meisten Nutzer nur grundlegende Bestandsverfolgungsfunktionen, während die fortgeschrittenen Fähigkeiten ungenutzt bleiben und Wartungs-Overhead schaffen. Das Unternehmen erkennt, dass es die Kernfunktionalität in sechs Monaten hätte ersetzen und fortgeschrittene Features inkrementell basierend auf tatsächlicher Nachfrage hätte hinzufügen können.
