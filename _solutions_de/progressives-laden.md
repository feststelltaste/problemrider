---
title: Progressives Laden
description: Inkrementelles Laden von Inhalten mit zunehmender Qualität.
category:
- Performance
problems:
- slow-application-performance
- poor-user-experience-ux-design
- high-client-side-resource-consumption
- user-frustration
- network-latency
- slow-response-times-for-lists
- high-resource-utilization-on-client
layout: solution
lang: de
en_slug: progressive-loading
related_solutions:
- slug: predictive-loading
  similarity: 0.8
- slug: lazy-loading
  similarity: 0.8
- slug: lazy-evaluation
  similarity: 0.75
- slug: predictive-prefetching
  similarity: 0.75
- slug: image-and-asset-optimization
  similarity: 0.7
- slug: performance-optimization
  similarity: 0.7
---

## Description

Progressives Laden liefert Inhalt in Stufen zunehmender Vollständigkeit oder Qualität — Text vor Bildern, eine niedrig aufgelöste Vorschau vor dem Vollqualitäts-Asset, eine Zusammenfassung vor dem vollständigen Detail —, sodass sofort etwas Bedeutsames auf dem Bildschirm erscheint, während der verbleibende, schwerere Inhalt im Hintergrund weiterlädt. Es beinhaltet typischerweise die Umstrukturierung von API-Antworten, sodass essenzielle Daten zuerst ankommen, die Nutzung von Platzhalter- oder Skeleton-UI, während vollständiger Inhalt aussteht, und die Priorisierung von Inhalt oberhalb der Bildschirmkante gegenüber Inhalt, zu dem der Nutzer noch nicht gescrollt hat. Dies ist speziell für Legacy-Systeme ein nützlicher Hebel, weil es wahrgenommene Performance adressiert, ohne jegliche Änderung am Backend zu erfordern, das tatsächlich die langsame Antwort produziert — ein Legacy-System, dessen Datenmodell oder Abfrageperformance teuer oder riskant anzufassen ist, kann sich dem Nutzer dennoch dramatisch schneller anfühlen, rein durch die Art, wie die bestehende, unveränderte Payload auf dem Client sequenziert und gerendert wird. Der Ansatz ist besonders effektiv bei langsamen Netzwerkverbindungen, wo eine einzelne große monolithische Antwort viele Sekunden braucht, um vollständig anzukommen, aber eine gestaffelte Lieferung dem Nutzer erlaubt, innerhalb eines Bruchteils einer Sekunde mit dem Lesen oder der Interaktion mit frühem Inhalt zu beginnen. Der Zielkonflikt ist zusätzliche Komplexität: Die Aufteilung der Inhaltslieferung in Stufen bedeutet insgesamt mehr Anfragen, potenzielle Layout-Verschiebungen, während späterer Inhalt ankommt, und eine größere Testoberfläche, da jede Ladestufe unabhängig verifiziert werden muss.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie Inhalt, der in Stufen geliefert werden kann: Text vor Bildern, niedrig aufgelöste Vorschauen vor Vollqualität, Zusammenfassung vor Detail
- Implementieren Sie Skeleton Screens oder Platzhalter-UI, die sofort rendern, während vollständiger Inhalt lädt
- Verwenden Sie progressive Bildformate (progressives JPEG, responsive Bilder), um niedrig aufgelöste Vorschauen anzuzeigen, die schärfer werden, während Daten ankommen
- Strukturieren Sie API-Antworten so, dass essenzielle Daten zuerst zurückgegeben werden, wobei ergänzende Daten über nachfolgende Anfragen geladen werden
- Priorisieren Sie das Laden von Inhalt oberhalb der Bildschirmkante und schieben Sie Inhalt unterhalb der Bildschirmkante auf, bis der Nutzer scrollt
- Wenden Sie progressive Verbesserung auf Legacy-Seiten an, indem Sie zuerst das Kern-HTML laden und anschließend mit JavaScript verbessern

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Reduziert die wahrgenommene Ladezeit, indem bedeutsamer Inhalt früh gezeigt wird
- Verbessert Nutzerengagement, indem leere Bildschirme während des Ladens verhindert werden
- Erlaubt Legacy-Systemen, sich reaktionsschnell anzufühlen, selbst mit langsamen Backends
- Funktioniert besonders gut bei langsamen Netzwerkverbindungen

**Kosten und Risiken:**
- Erfordert die Umstrukturierung, wie Inhalt geliefert wird, was in Legacy-Architekturen komplex sein kann
- Mehrere Ladestufen erhöhen die Anzahl der Anfragen, was die Gesamtladezeit potenziell erhöht
- Layout-Verschiebungen während progressiven Renderns können Nutzer desorientieren, wenn nicht sorgfältig gehandhabt
- Testing wird komplexer, da jede Ladestufe unabhängig verifiziert werden muss
- Inhaltsprioritätsentscheidungen könnten nicht mit allen Nutzer-Arbeitsabläufen übereinstimmen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-Immobilienangebotsplattform lieferte hochauflösende Immobilienbilder und detaillierte Angebotsdaten in einer einzigen großen Antwort, was 6-sekündige Ladezeiten bei typischen Verbindungen verursachte. Das Team strukturierte die Seite um, um sofort Angebotstext und einen niedrig aufgelösten Thumbnail anzuzeigen, dann progressiv die vollständige Bildgalerie und Nachbarschaftsanalysen zu laden. Der Angebotstext erschien innerhalb von 800 Millisekunden, was Nutzern etwas zum Lesen gab, während der schwerere Inhalt im Hintergrund lud. Diese Änderung reduzierte die Absprungrate um 25 Prozent, ohne jegliche Änderungen am Backend-Datenmodell zu erfordern.
