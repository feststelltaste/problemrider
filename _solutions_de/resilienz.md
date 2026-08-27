---
title: Resilienz
description: Fähigkeit eines Systems, unter widrigen Bedingungen oder
  Fehlern funktionsfähig zu bleiben.
category:
- Architecture
problems:
- cascade-failures
- system-outages
- unpredictable-system-behavior
- brittle-codebase
- single-points-of-failure
- fear-of-change
- constant-firefighting
layout: solution
lang: de
en_slug: resilience
related_solutions:
- slug: chaos-engineering
  similarity: 0.85
- slug: failover-mechanisms
  similarity: 0.8
- slug: retry
  similarity: 0.8
- slug: redundancy
  similarity: 0.8
- slug: secure-software
  similarity: 0.75
- slug: incident-management
  similarity: 0.75
---

## Description

Resilienz beschreibt die Fähigkeit eines Systems, weiterzuarbeiten, in einem möglicherweise degradierten, aber noch nützlichen Modus, wenn Teile davon ausfallen oder sich seine Umgebung unerwartet verhält, statt beim ersten Fehler vollständig zusammenzubrechen. Sie wird durch eine Kombination spezifischer Muster erreicht — Circuit Breaker, die aufhören, eine fehlschlagende Abhängigkeit aufzurufen, Timeouts, die begrenzen, wie lange eine Komponente auf eine Antwort wartet, Bulkheads, die Ressourcenpools isolieren, sodass eine überlastete Komponente nicht andere aushungern kann, und Redundanz, die einen alternativen Pfad bietet, wenn ein primärer nicht verfügbar ist. Legacy-Systeme sind konstruktionsbedingt häufig das Gegenteil von resilient: Komponenten wurden unter der Annahme gebaut, dass ihre Abhängigkeiten immer verfügbar sein und immer schnell antworten würden, sodass ein einzelner langsamer oder fehlschlagender Dienst dazu neigt, zu einem totalen Ausfall zu kaskadieren statt zu einem eingedämmten, teilweisen. Resilienzmuster nachträglich an den Integrationspunkten eines Legacy-Systems einzubauen ist üblicherweise weit handhabbarer als das Innenleben des Systems neu zu schreiben, weil die Muster bestehende Aufrufe umhüllen, statt zu erfordern, zu verstehen, was in ihnen passiert. Dies macht Resilienz zu einer besonders praktischen Investition während der Modernisierung, da sie direkt den Explosionsradius genau der Fehler reduziert, die inkrementelle Legacy-Änderungen am wahrscheinlichsten einführen, und sie baut das organisatorische Vertrauen auf, das nötig ist, um das System weiter zu ändern, ohne Angst, dass ein einzelner Fehler die ganze Plattform zu Fall bringt.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Bewerten Sie die aktuelle Resilienzlage, indem Sie Fehlermodi und ihre Auswirkung auf das Legacy-System katalogisieren
- Implementieren Sie Circuit Breaker, Wiederholungen mit exponentiellem Backoff und Timeouts an allen Integrationspunkten
- Gestalten Sie für Teilverfügbarkeit, sodass Fehler in nicht kritischen Komponenten die Kernfunktionalität nicht beeinträchtigen
- Fügen Sie Bulkheads hinzu, um Fehler zu isolieren und zu verhindern, dass Ressourcenerschöpfung sich über Komponenten hinweg ausbreitet
- Führen Sie Chaos-Engineering-Experimente durch, um unbekannte Fehlermodi zu entdecken und zu adressieren
- Bauen Sie Redundanz in kritische Pfade ein und stellen Sie sicher, dass Failover-Mechanismen regelmäßig getestet werden
- Erstellen und pflegen Sie Runbooks für bekannte Fehlerszenarien, um schnelle Wiederherstellung zu ermöglichen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Das System bedient Nutzer weiterhin während teilweiser Ausfälle, statt totale Ausfälle zu erleben
- Reduziert die geschäftliche Auswirkung von Infrastruktur- und Softwareausfällen
- Baut Teamvertrauen auf, Änderungen vorzunehmen, im Wissen, dass das System Fehler tolerieren kann
- Bietet einen systematischen Ansatz zur schrittweisen Verbesserung der Legacy-System-Zuverlässigkeit

**Kosten und Risiken:**
- Resilienzmuster fügen bereits komplexen Legacy-Codebasen Komplexität hinzu
- Das Testen von Resilienzmechanismen erfordert dedizierten Aufwand und Tooling
- Überkonstruierte Resilienz für nicht kritische Komponenten verschwendet Ressourcen
- Resilienzmechanismen selbst können ausfallen oder unerwartetes Verhalten verursachen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine große E-Commerce-Plattform erlebte kaskadierende Ausfälle jedes Mal, wenn ihr Legacy-Bestandsdienst langsam wurde, weil jeder andere Dienst unbegrenzt auf Bestandsantworten wartete. Durch systematisches Hinzufügen von Circuit Breakern, Timeouts und Fallback-Verhalten an jedem Integrationspunkt verwandelte das Team das System von einem, bei dem jeder Dienstausfall totalen Zusammenbruch verursachte, in eines, bei dem Ausfälle eingedämmt waren und Nutzer nur geringfügige Feature-Verschlechterung erlebten. Der Bestandsdienst konnte nun bereitgestellt, neu gestartet oder sogar vollständig ausfallen, ohne den Checkout-Ablauf zu beeinträchtigen, der zwischengespeicherte Bestandsdaten nutzte, wenn der Live-Dienst nicht verfügbar war.
