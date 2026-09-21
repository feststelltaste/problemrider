---
title: Häufige Anforderungsänderungen
description: Die Anforderungen eines Projekts oder Features werden ständig aktualisiert,
  selbst nachdem die Entwicklung begonnen hat, was zu Nacharbeit, Verzögerungen und
  Frustration führt.
category:
- Communication
- Process
related_problems:
- slug: constantly-shifting-deadlines
  similarity: 0.8
- slug: changing-project-scope
  similarity: 0.75
- slug: stakeholder-developer-communication-gap
  similarity: 0.7
- slug: scope-creep
  similarity: 0.7
- slug: no-formal-change-control-process
  similarity: 0.7
- slug: large-estimates-for-small-changes
  similarity: 0.7
solutions:
- evolutionary-requirements-development
- formal-change-control-process
- product-owner
- requirements-analysis
- security-requirements-definition
- definition-of-ready
- regular-stakeholder-demonstrations
- domain-immersion
- story-mapping
- specification-by-example
layout: problem
lang: de
en_slug: frequent-changes-to-requirements
---

## Description
Häufige Anforderungsänderungen treten auf, wenn sich Umfang und Spezifikationen eines Projekts in ständigem Fluss befinden, selbst nachdem die Entwicklung bereits läuft. Dies ist mehr als nur agile Anpassung; es ist ein Anzeichen für Instabilität im Fundament des Projekts. Wenn Anforderungen nicht vorab gut definiert oder vereinbart sind, werden Teams gezwungen, ständig umzuschwenken, was zu verschwendeter Arbeit, verpassten Terminen und sinkender Teammoral führt. Dieses Problem verweist oft auf tiefere Probleme bei Kommunikation, Planung und Stakeholder-Abstimmung.

## Indicators ⟡
- Der Umfang des Projekts weitet sich ständig aus.
- Das Team verpasst häufig Termine.
- Das Team wechselt ständig den Kontext.
- Es gibt viel Nacharbeit.

## Symptoms ▲

- [Implementierungs-Nacharbeit](implementierungs-nacharbeit.md)
<br/>  Features müssen neu gebaut werden, wenn sich Anforderungen ändern, nachdem die Entwicklung bereits begonnen hat, was bisherigen Aufwand verschwendet.
- [Verpasste Termine](verpasste-termine.md)
<br/>  Ständige Anforderungsänderungen zwingen das Team, Arbeit zu wiederholen, was dazu führt, dass Projekte ihre geschätzten Zeitpläne durchgängig überschreiten.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Ständiges Umschwenken und Wiederholen von Arbeit demoralisiert Entwickler und führt zu Frustration und Erschöpfung.
- [Verschwendeter Entwicklungsaufwand](verschwendeter-entwicklungsaufwand.md)
<br/>  Arbeit, die gegen frühere Anforderungen abgeschlossen wurde, wird obsolet, wenn sich Anforderungen ändern, was verschwendeten Aufwand darstellt.
- [Scope Creep](scope-creep.md)
<br/>  Häufige Anforderungsänderungen weiten oft den Gesamtprojektumfang über die ursprünglichen Pläne hinaus aus.
- [Teamverwirrung](teamverwirrung.md)
<br/>  Ständige Änderungen lassen Teammitglieder im Unklaren über aktuelle Anforderungen und Prioritäten.

## Causes ▼

- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Schlecht erhobene Anfangsanforderungen benötigen häufige Korrekturen, während Lücken und Missverständnisse während der Entwicklung entdeckt werden.
- [Kommunikationslücke zwischen Stakeholdern und Entwicklern](kommunikationsluecke-zwischen-stakeholdern-und-entwicklern.md)
<br/>  Missverständnisse zwischen Stakeholdern und Entwicklern führen dazu, dass Anforderungen überarbeitet werden, während die tatsächlichen Bedürfnisse klarer werden.
- [Chaos in der Produktrichtung](chaos-in-der-produktrichtung.md)
<br/>  Widersprüchliche Prioritäten mehrerer Stakeholder ohne klare Produktführung verursachen häufige Verschiebungen der Anforderungen.
- [Marktdruck](marktdruck.md)
<br/>  Externe Wettbewerbskräfte treiben plötzliche Änderungen der Geschäftsstrategie an, die in geänderte Anforderungen münden.

## Detection Methods ○

- **Analyse des Versionskontrollsystems:** Nachverfolgung von Änderungen an Anforderungsdokumenten oder User Stories im Projektmanagement-Tool.
- **Projektmanagement-Metriken:** Beobachtung von Änderungen im Projektumfang, geschätzten vs. tatsächlichen Fertigstellungszeiten und der Anzahl wiedereröffneter Aufgaben.
- **Team-Retrospektiven:** Diskussion wiederkehrender Probleme im Zusammenhang mit sich ändernden Anforderungen und ihrer Auswirkung auf das Team.
- **Stakeholder-Interviews:** Befragung von Stakeholdern zu ihrem Vertrauen in die aktuellen Anforderungen und ihrem Verständnis des Entwicklungsprozesses.

## Examples
Ein Team für die Entwicklung mobiler Apps ist auf halbem Weg beim Bau eines neuen Nutzerprofil-Bildschirms, als die Marketingabteilung entscheidet, dass sie ein völlig anderes Layout und zusätzliche Felder für eine neue Kampagne benötigt. Die Entwickler müssen einen Großteil ihrer Arbeit verwerfen und von vorn beginnen. Ähnlich wird während der Entwicklung einer API das Datenmodell ständig vom Product Owner auf Basis neuer Erkenntnisse aus der Nutzerforschung überarbeitet, was häufige Datenbankschema-Migrationen und Code-Refactoring erzwingt. Dieses Problem ist eine klassische Herausforderung in der Softwareentwicklung, die oft aus einer Trennung zwischen Geschäftsstrategie und Umsetzung entsteht. Während manche Änderungen unvermeidlich sind, können häufige, ungeplante Änderungen den Fortschritt eines Projekts und die Teammoral lähmen.
