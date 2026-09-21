---
title: Performance-Optimierung
description: Verbesserung der wahrgenommenen Reaktionsfähigkeit durch
  nutzerseitige Performance-Techniken.
category:
- Performance
quality_tactics_url: https://qualitytactics.de/en/usability/performance-optimization/
problems:
- slow-application-performance
- user-frustration
- poor-user-experience-ux-design
- slow-response-times-for-lists
- high-client-side-resource-consumption
- negative-user-feedback
- gradual-performance-degradation
- customer-dissatisfaction
layout: solution
lang: de
en_slug: performance-optimization
related_solutions:
- slug: predictive-loading
  similarity: 0.8
- slug: caching-strategy
  similarity: 0.8
- slug: asynchronous-operations
  similarity: 0.8
- slug: lazy-loading
  similarity: 0.75
- slug: efficient-algorithms
  similarity: 0.75
- slug: performance-budgets
  similarity: 0.75
---

## Description

Nutzerseitige Performance-Optimierung zielt auf das, was eine Person tatsächlich erlebt — Skeleton Screens, virtuelles Scrollen, aufgeschobenes Laden nicht kritischer Inhalte —, statt auf rohe serverseitige Metriken, die die wahrgenommene Geschwindigkeit möglicherweise überhaupt nicht widerspiegeln. Legacy-Oberflächen neigen besonders zu vollständigen Seiten-Reloads und Warten vor leerem Bildschirm, beides fühlt sich weit langsamer an, als es dieselbe Backend-Antwortzeit tun würde, wenn die Oberfläche irgendeinen Hinweis darauf gäbe, dass etwas passiert. Da diese Techniken einen Bildschirm nach dem anderen angewendet werden können, ohne das Backend anzufassen, sind sie oft der schnellste Weg, ein trödeliges Legacy-System modern wirken zu lassen, obwohl ein echtes Risiko besteht, dass diese Fixes ein zugrunde liegendes Backend-Problem verdecken, das tatsächlich behoben werden muss, statt nur wahrnehmungsmäßig geglättet zu werden.

## How to Apply ◆

> Legacy-Systeme leiden oft unter langsamer wahrgenommener Performance aufgrund vollständiger Seiten-Reloads, unoptimierten Renderings und blockierender Operationen. Nutzerseitige Performance-Optimierung zielt auf das, was Nutzer tatsächlich erleben, statt auf rohe Server-Metriken.

- Implementieren Sie Skeleton Screens und Lade-Platzhalter, die die Struktur der Seite sofort zeigen, während Daten laden. Dies lässt die Anwendung schneller wirken, als einen leeren Bildschirm oder Spinner zu zeigen.
- Fügen Sie Paginierung oder virtuelles Scrollen zu großen Datensätzen hinzu. Legacy-Systeme, die Tausende von Zeilen gleichzeitig in den Browser laden, verursachen erhebliche Render-Verzögerungen und exzessiven Speicherverbrauch.
- Schieben Sie das Laden nicht kritischer Inhalte auf. Wenden Sie Lazy Loading auf Bilder, sekundäre Datenpanels und Inhalte unterhalb der Bildschirmkante an, sodass die primäre Ansicht schnell rendert.
- Optimieren Sie den kritischen Rendering-Pfad, indem Sie essenzielles CSS und JavaScript zuerst laden und nicht essenzielle Ressourcen aufschieben. Legacy-Systeme laden oft große monolithische Bundles, die das Rendering blockieren.
- Implementieren Sie clientseitiges Caching für Daten, die sich selten ändern, um die Anzahl der Server-Roundtrips zu reduzieren und Antwortzeiten für wiederholte Ansichten zu verbessern.
- Prefetchen Sie Daten für wahrscheinliche nächste Aktionen basierend auf Nutzerverhaltensmustern. Wenn Nutzer typischerweise von einer Liste zu einer Detailansicht navigieren, beginnen Sie mit dem Laden der Detaildaten, sobald der Nutzer über ein Listenelement hovert oder es fokussiert.

## Tradeoffs ⇄

> Verbesserungen der wahrgenommenen Performance lassen das System dramatisch schneller wirken, fügen aber Frontend-Komplexität hinzu und können zugrunde liegende Backend-Performance-Probleme verdecken.

**Vorteile:**

- Verbessert die Nutzerwahrnehmung der Anwendungsgeschwindigkeit dramatisch, selbst wenn Backend-Verarbeitungszeiten unverändert bleiben.
- Reduziert Nutzerfrustration durch das Warten auf langsam ladende Seiten, was eine der häufigsten Beschwerden über Legacy-Systeme ist.
- Verringert clientseitigen Ressourcenverbrauch durch effiziente Render- und Datenladestrategien.
- Kann inkrementell auf einzelnen Bildschirmen implementiert werden, ohne eine vollständige Frontend-Neuschreibung zu erfordern.

**Kosten und Risiken:**

- Skeleton Screens und optimistische Updates können Nutzer irreführen, wenn die tatsächlichen Daten erheblich länger zum Laden brauchen oder von der Vorschau abweichen.
- Clientseitiges Caching führt das Risiko ein, veraltete Daten anzuzeigen, was sorgfältige Cache-Invalidierungsstrategien erfordert.
- Die Optimierung wahrgenommener Performance kann den Druck reduzieren, zugrunde liegende Backend-Performance-Probleme zu beheben, was ihnen erlaubt, sich über die Zeit zu verschlechtern.
- Die Implementierung von Lazy Loading und virtuellem Scrollen in Legacy-Frontend-Frameworks, die dies nicht nativ unterstützen, kann technisch anspruchsvoll sein.

## How It Could Be

> Nutzer beurteilen die Anwendungsgeschwindigkeit nach dem, was sie sehen, nicht nach dem, was die Server-Logs sagen. Die Optimierung der wahrgenommenen Performance kann ein trödeliges Legacy-System transformieren, ohne das Backend anzufassen.

Ein Legacy-Kundenverwaltungssystem lädt eine Kundenlistenseite, die alle Kundendatensätze aus der Datenbank abruft und sie in einer einzigen HTML-Tabelle rendert. Mit über fünfzigtausend Kunden dauert das Laden der Seite zwölf Sekunden, während derer Nutzer einen leeren weißen Bildschirm sehen. Das Team implementiert drei Änderungen: virtuelles Scrollen, das nur die sichtbaren Zeilen rendert, einen Skeleton Screen, der sofort erscheint, während Daten laden, und paginierte API-Aufrufe, die jeweils hundert Datensätze abrufen. Der erste sinnvolle Inhalt erscheint nun in unter einer Sekunde, und Nutzer können sofort mit Scrollen und Suchen beginnen, während zusätzliche Daten im Hintergrund laden. Obwohl die gesamte Datenübertragung gleich bleibt, nehmen Nutzer das System als dramatisch schneller wahr, weil sie fast sofort mit der Arbeit beginnen können, statt auf einen leeren Bildschirm zu starren.
