---
title: Kognitive Überlastung
description: Entwickler müssen zu viele komplexe Systeme oder Konzepte gleichzeitig
  im Arbeitsgedächtnis behalten, was ihre Effektivität verringert.
category:
- Code
- Culture
- Process
related_problems:
- slug: increased-cognitive-load
  similarity: 0.85
- slug: mental-fatigue
  similarity: 0.75
- slug: context-switching-overhead
  similarity: 0.7
- slug: avoidance-behaviors
  similarity: 0.7
- slug: developer-frustration-and-burnout
  similarity: 0.65
- slug: difficult-developer-onboarding
  similarity: 0.65
solutions:
- clean-code
- design-by-contract
- loose-coupling
- separation-of-concerns
- cognitive-load-minimization
- form-design
- intuitive-navigation
- progressive-disclosure
- search-function
- visual-hierarchy
layout: problem
lang: de
en_slug: cognitive-overload
---

## Description

Kognitive Überlastung entsteht, wenn Entwickler mehr komplexe Informationen verstehen und damit arbeiten müssen, als effektiv im Arbeitsgedächtnis gehalten werden können. Dies geschieht, wenn Systeme übermäßig komplex sind, wenn Entwickler gleichzeitig über mehrere Domänen hinweg arbeiten müssen, oder wenn die Architektur das Verständnis vieler miteinander verbundener Komponenten für einfache Änderungen erfordert. Das menschliche Gehirn hat eine begrenzte Kapazität des Arbeitsgedächtnisses, und das Überschreiten dieser Kapazität führt zu verringerter Leistung, erhöhten Fehlern und mentaler Erschöpfung.

## Indicators ⟡

- Entwickler verlieren häufig den Überblick darüber, woran sie gearbeitet haben
- Einfache Aufgaben erfordern umfangreiches Notieren oder Dokumentation, um sie abzuschließen
- Teammitglieder äußern das Gefühl, von der Systemkomplexität überwältigt zu sein
- Entwickler vermeiden es, an bestimmten Teilen des Systems zu arbeiten, aufgrund der Komplexität
- Häufige Fehler entstehen durch das Vergessen wichtigen Kontexts oder wichtiger Einschränkungen

## Symptoms ▲

- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Wenn Entwickler die Kapazität des Arbeitsgedächtnisses überschreiten, übersehen sie wichtige Einschränkungen und führen Defekte ein.
- [Vermeidungsverhalten](vermeidungsverhalten.md)
<br/>  Entwickler schieben kognitiv anspruchsvolle Teile des Systems auf oder vermeiden sie.
- [Mentale Erschöpfung](mentale-erschoepfung.md)
<br/>  Anhaltende kognitive Überlastung führt zu Erschöpfung und verringerter Fähigkeit, produktive Arbeit zu leisten.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Das Verständnis komplexer, miteinander verbundener Systeme vor Änderungen verlangsamt die Feature-Umsetzung.
- [Verringerte individuelle Produktivität](verringerte-individuelle-produktivitaet.md)
<br/>  Entwickler erledigen weniger Aufgaben, weil jede Änderung das Verständnis weit mehr Kontexts erfordert als die Änderung selbst.
- [Prokrastination bei komplexen Aufgaben](prokrastination-bei-komplexen-aufgaben.md)
<br/>  Überwältigte Entwickler schieben kognitiv anspruchsvolle Aufgaben zugunsten einfacherer, weniger wirkungsvoller Arbeit auf.

## Causes ▼

- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Stark gekoppelte Systeme erfordern das Verständnis vieler miteinander verbundener Komponenten für selbst einfache Änderungen.
- [Komplexe und unklare Logik](komplexe-und-unklare-logik.md)
<br/>  Code, der schwer zu lesen und zu verstehen ist, zwingt Entwickler dazu, übermäßigen mentalen Aufwand für das Verständnis aufzubringen.
- [Overhead durch Kontextwechsel](overhead-durch-kontextwechsel.md)
<br/>  Häufiger Wechsel zwischen unterschiedlichen Systemen oder Problemdomänen zersplittert die Aufmerksamkeit und überlastet das Arbeitsgedächtnis.
- [Komplexes Domänenmodell](komplexes-domaenenmodell.md)
<br/>  Inhärent komplexe Geschäftsdomänen erfordern, dass Entwickler umfangreiches Domänenwissen im Arbeitsgedächtnis halten.
- [Spaghetticode](spaghetticode.md)
<br/>  Spaghetticode mit verworrenem, unstrukturiertem Kontrollfluss zwingt Entwickler dazu, komplexe Ausführungspfade nachzuverfolgen, was direkt zur kognitiven Überlastung beiträgt.

## Detection Methods ○

- **Komplexitätsmetriken:** Messung zyklomatischer Komplexität, Kopplung und anderer architektonischer Komplexitätsindikatoren
- **Entwickler-Umfragen:** Befragung von Teammitgliedern zu kognitiver Belastung und mentaler Arbeitslast
- **Fehlerraten-Analyse:** Beobachtung der Korrelation zwischen Systemkomplexität und Häufigkeit von Entwicklerfehlern
- **Aufgabenerledigungszeit-Tracking:** Vergleich der Erledigungszeiten für Aufgaben ähnlichen Umfangs, aber unterschiedlicher Komplexität
- **Fokuszeit-Analyse:** Messung, wie lange Entwickler den Fokus auf komplexe Aufgaben aufrechterhalten können

## Examples

Ein Entwickler, der an einer E-Commerce-Plattform arbeitet, muss gleichzeitig die Struktur des Produktkatalogs, Regeln der Bestandsverwaltung, Preisalgorithmen, Steuerberechnungslogik, Bestimmung der Versandkosten und Systeme zur Aktionsbehandlung verstehen, um ein einfaches "Jetzt kaufen"-Feature umzusetzen. Die Verbindungen zwischen diesen Systemen erfordern die Aufrechterhaltung detaillierter mentaler Modelle jeder Komponente, was die kognitive Kapazität übersteigt und zu Fehlern in der Implementierung führt. Ein weiteres Beispiel betrifft einen Entwickler, der ein Finanzhandelssystem ändert, bei dem das Verständnis einer einzelnen Funktion Kenntnisse über Marktdatenprotokolle, Risikomanagementregeln, regulatorische Compliance-Anforderungen, Portfolio-Optimierungsalgorithmen und Echtzeit-Ereignisverarbeitungsmuster erfordert, was kognitive Überlastung erzeugt, die selbst einfache Änderungen fehleranfällig und zeitaufwendig macht.
