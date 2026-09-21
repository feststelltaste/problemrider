---
title: Fehlpassung zwischen Prozess und Software
description: Die Software wurde verbogen, um zu einem historisch gewachsenen Prozess
  zu passen, statt den Prozess gegen das zu prüfen, was das Produkt annimmt.
category:
- Business
- Process
- Requirements
related_problems:
- slug: reimplemented-standard-functionality
  similarity: 0.65
- slug: excessive-customization
  similarity: 0.65
- slug: core-modification-of-standard-software
  similarity: 0.6
- slug: process-design-flaws
  similarity: 0.6
- slug: customization-outside-version-control
  similarity: 0.6
- slug: architectural-mismatch
  similarity: 0.6
solutions:
- fit-to-standard-principle
- domain-immersion
- business-process-modeling
- functional-gap-analysis
- regular-stakeholder-demonstrations
- outcome-based-goal-setting
- executive-sponsorship
- value-stream-mapping
- pilot-projects
- definition-of-ready
layout: problem
lang: de
en_slug: process-software-misfit
---

## Description

Fehlpassung zwischen Prozess und Software tritt auf, wenn ein kommerzielles Softwareprodukt angepasst wird, um die bestehende Arbeitsweise einer Organisation zu reproduzieren, ohne dass jemand fragt, ob diese Arbeitsweise es wert ist, bewahrt zu werden. Kommerzielle Software kodiert ein Prozessmodell, und ein Großteil ihres Werts kommt daher, dass dieses Modell kohärent ist und über viele Kunden hinweg verfeinert wurde. Eine Organisation, die das Modell überschreibt, um ihrer eigenen historisch gewachsenen Praxis zu entsprechen, bezahlt für das Produkt, verwirft die darin enthaltene Argumentation und übernimmt die Kosten, den Unterschied für immer zu warten. Die Fehlpassung ist selten eine bewusste Entscheidung. Sie folgt aus einem Anforderungsprozess, der aufzeichnet, wie Dinge heute gemacht werden, und das als Spezifikation behandelt, sowie aus der Tatsache, dass niemand dafür verantwortlich ist, die Arbeitsweise des Geschäfts zu ändern.

## Indicators ⟡

- Anforderungen wurden durch Dokumentation des aktuellen Prozesses erhoben und nicht hinterfragt
- Die Anpassungsliste wird von Elementen dominiert, die bestehende Schritte reproduzieren, statt neue Ergebnisse zu ermöglichen
- Nutzer beschreiben das neue System als „wie das alte, aber langsamer"
- Niemand kann erklären, warum ein Schritt existiert, außer dass er schon immer da war
- Standard-Produktschulung wird nicht genutzt, weil sie nicht beschreibt, wie Ihre Installation funktioniert
- Prozessänderungen werden durch explizite Vereinbarung als außerhalb des Umfangs des Softwareprojekts betrachtet

## Symptoms ▲

- [Übermäßige Anpassung](uebermaessige-anpassung.md)
<br/>  Die Reproduktion eines bestehenden Prozesses in einem Produkt, das um einen anderen herum gebaut ist, erfordert Anpassung an jedem Punkt, an dem sie sich unterscheiden.
- [Ineffiziente Prozesse](ineffiziente-prozesse.md)
<br/>  Historische Ineffizienzen werden bewahrt und kodiert, und danach sind sie schwerer zu entfernen als vor der Automatisierung.
- [Neu implementierte Standardfunktionalität](neu-implementierte-standardfunktionalitaet.md)
<br/>  Wo sich die Version eines Schritts im Produkt von der lokalen unterscheidet, wird stattdessen die lokale Version gebaut, statt den Standard zu übernehmen.
- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Aufwand wird darauf verwendet, das nachzubauen, was die Organisation bereits hatte, statt auf Fähigkeiten, die sie nicht hatte.
- [Schwierigkeiten beim Quantifizieren von Nutzen](schwierigkeiten-beim-quantifizieren-von-nutzen.md)
<br/>  Ein Projekt, das den vorherigen Prozess reproduziert, liefert wenig messbare Verbesserung, was es danach schwer macht, seinen Wert zu demonstrieren.
- [Nutzerfrustration](nutzerfrustration.md)
<br/>  Nutzer erleben die Störung eines neuen Systems ohne den Vorteil eines besseren Prozesses, was die am wenigsten günstige Kombination ist.
- [Upgrade durch Anpassungen blockiert](upgrade-durch-anpassungen-blockiert.md)
<br/>  Der angehäufte Unterschied zwischen dem lokalen Prozess und dem Modell des Produkts muss durch jedes nachfolgende Release mitgeführt werden.

## Causes ▼

- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Die Dokumentation des aktuellen Zustands wird als Anforderungsanalyse behandelt, sodass die Spezifikation eine Beschreibung der Vergangenheit ist.
- [Vakuum an Projektautorität](vakuum-an-projektautoritaet.md)
<br/>  Die Änderung der Arbeitsweise einer Abteilung erfordert Autorität, die das Projekt nicht hat, sodass stattdessen die Software geändert wird.
- [Übermäßige Gefälligkeit gegenüber Stakeholdern](uebermaessige-gefaelligkeit-gegenueber-stakeholdern.md)
<br/>  Abteilungen, gefragt, was sie brauchen, beschreiben, was sie jetzt tun, und die Antworten werden akzeptiert statt geprüft.
- [Mangel an Legacy-Fachkräften](mangel-an-legacy-fachkraeften.md)
<br/>  Niemand Beteiligtes kennt das Prozessmodell des Produkts gut genug, um zu argumentieren, dass der Standard dienen würde, sodass der lokale Prozess standardmäßig gewinnt.
- [Abhängigkeit vom Zulieferer](abhaengigkeit-vom-zulieferer.md)
<br/>  Ein für die Entwicklung bezahlter Implementierungspartner hat keinen Anreiz zu argumentieren, dass der Kunde stattdessen seinen Prozess ändern sollte.
- [Marktdruck](marktdruck.md)
<br/>  Prozessänderung dauert länger als Softwareänderung, und wo ein Termin festgelegt ist, wird die schnellere Option gewählt, unabhängig davon, welche besser ist.

## Detection Methods ○

- Überprüfung der Anforderungen aus der letzten Implementierung und Zählung, wie viele aktuelle Praxis versus ein gewünschtes Ergebnis beschreiben
- Für jede bedeutende Anpassung fragen, was der Standard getan hätte und warum er abgelehnt wurde; fehlende Antworten deuten darauf hin, dass die Frage nie gestellt wurde
- Vergleich von Prozessmetriken vor und nach der Implementierung; ein Projekt, das den Prozess reproduziert hat, wird wenig Bewegung zeigen
- Fragen von Nutzern, ob das System so funktioniert, wie es die Schulung des Anbieters beschreibt, und wo es abweicht
- Suche nach Schritten im Prozess, die nur existieren, weil ein vorheriges System sie erforderte
- Prüfen, ob das Projekt ein Mandat hatte, Geschäftsprozesse zu ändern, und ob es genutzt wurde

## Examples

Ein Logistikunternehmen implementierte ein Lagerverwaltungsprodukt und passte 40 Bereiche an, um zu entsprechen, wie ihre Standorte bereits arbeiteten. Zwei Jahre später verglich eine externe Überprüfung ihren Kommissionierprozess mit dem Standardmodell des Produkts und stellte fest, dass sechs der Anpassungen Schritte bewahrten, die in den 1990er-Jahren eingeführt wurden, um ein papierbasiertes System zu kompensieren, das 2004 abgeschafft worden war. Ein Schritt erforderte eine Vorgesetzten-Gegenzeichnung für eine Bewegung, die das Produkt von vornherein daran gehindert hätte, inkorrekt zu sein. Die Entfernung der sechs Anpassungen und die Übernahme des Standardprozesses reduzierte die Kommissionierzeit pro Bestellung um etwa elf Prozent – eine Verbesserung, die am ersten Tag der Implementierung verfügbar gewesen war und wegangepasst worden war.

Das Autoritätsproblem war die zugrunde liegende Ursache und war in den eigenen Aufzeichnungen des Projekts sichtbar. Die Projektcharta stellte explizit fest, dass das Projekt keine Änderungen an Lagerbetriebsverfahren erfordern würde, mit der Begründung, dass der Betrieb während der Hochsaison keine Störung absorbieren könnte. Diese Einschränkung wurde für eine Saison festgelegt und wurde über eine zweijährige Implementierung nie überprüft. Niemand Beteiligtes hatte die Position, sie wieder zu öffnen, und als die Überprüfung stattfand, waren die Anpassungen lange genug in Produktion, dass ihre Entfernung selbst eine Änderung war, die die Autorität erforderte, die zu Beginn gefehlt hatte.
