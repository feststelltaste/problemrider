---
title: Graceful Degradation
description: Fähigkeit eines Systems, bei Ausfällen oder Überlast in eingeschränkter
  Kapazität weiterzuarbeiten.
category:
- Architecture
- Operations
problems:
- system-outages
- cascade-failures
- unpredictable-system-behavior
- slow-application-performance
- capacity-mismatch
- constant-firefighting
- customer-dissatisfaction
- upstream-timeouts
layout: solution
lang: de
en_slug: graceful-degradation
related_solutions:
- slug: resilience
  similarity: 0.75
- slug: failover-mechanisms
  similarity: 0.75
- slug: rate-limiting
  similarity: 0.75
- slug: load-shedding
  similarity: 0.75
- slug: redundancy
  similarity: 0.7
- slug: rollback-mechanisms
  similarity: 0.7
---

## Description

Graceful Degradation ist die Designeigenschaft, die es einem System erlaubt, seine wichtigsten Funktionen in reduzierter Kapazität weiter zu erbringen, wenn Teile davon ausfallen oder überlastet werden, statt vollständig zu versagen. Sie funktioniert, indem Features nach geschäftlicher Kritikalität klassifiziert werden, Fallback-Verhalten für die weniger kritischen definiert wird — zwischengespeicherte Daten, vereinfachte Antworten, deaktivierte nicht essenzielle Features — und Überlast oder Teilausfall früh genug erkannt wird, um in einen degradierten Modus zu wechseln, bevor das System einen harten Ausfallschwellenwert erreicht. Dies unterscheidet sich von Redundanz oder Failover, die darauf abzielen, volle Funktionalität verfügbar zu halten, indem ein Ausfall vollständig maskiert wird; Graceful Degradation akzeptiert stattdessen eine sichtbare, kontrollierte Reduzierung des Dienstes als die bewusste Alternative zu einem unkontrollierten Ausfall. Legacy-Systeme sind häufig anfällig für vollständige Ausfälle genau deshalb, weil Komponenten, die unabhängig sein sollten — eine Empfehlungs-Engine und der Checkout-Flow, zum Beispiel — Ressourcen oder Fail-Fast-Codepfade teilen, die nie mit Isolation im Blick entworfen wurden, sodass ein Lastspitze auf einem Randfeature die gesamte Anwendung lahmlegen kann. Graceful Degradation in ein solches System einzuführen bedeutet, Grenzen um nicht essenzielle Funktionalität nachzurüsten, sodass ihr Ausfall oder ihre Drosselung nicht in Kernworkflows kaskadieren kann, und verwandelt so, was sonst ein vollständiger Ausfall wäre, in einen vorübergehenden, eingedämmten Verlust von Sekundärfunktionalität, den die meisten Nutzer vielleicht nicht einmal bemerken.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Klassifizieren Sie Systemfeatures nach geschäftlicher Kritikalität, um zu bestimmen, welche degradiert werden können und welche vollständig verfügbar bleiben müssen
- Implementieren Sie Fallback-Verhalten für nicht kritische Features (zwischengespeicherte Daten, vereinfachte Antworten, statischer Inhalt)
- Fügen Sie Lasterkennungslogik hinzu, die Degradationsmodi aktiviert, bevor das System harte Ausfallschwellenwerte erreicht
- Gestalten Sie Degradation transparent für Nutzer, indem angemessene Meldungen über reduzierte Funktionalität angezeigt werden
- Testen Sie Degradationsmodi regelmäßig, um sicherzustellen, dass Fallback-Pfade tatsächlich funktionieren, wenn sie gebraucht werden
- Nutzen Sie Feature Toggles oder Konfigurations-Flags, um Degradation manuell während erwarteter Hochlast-Ereignisse auszulösen
- Überwachen Sie Degradationszustandsübergänge und alarmieren Sie Betriebsteams, wenn das System in den reduzierten Modus eintritt

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Kernfunktionalität bleibt selbst bei Teilausfällen oder Überlast verfügbar
- Verringert Häufigkeit und Schweregrad vollständiger Systemausfälle
- Bietet eine bessere Nutzererfahrung als harte Fehlschläge oder Fehlerseiten
- Verschafft Betriebsteams Zeit, zugrundeliegende Probleme anzugehen

**Kosten und Risiken:**
- Der Entwurf und die Pflege von Fallback-Pfaden fügt Entwicklungs- und Testaufwand hinzu
- Nutzer könnten nicht bemerken, dass sie degradierte Funktionalität erhalten, was zu Dateninkonsistenzen führt
- Degradationslogik kann systemische Probleme verdecken, die sich über die Zeit verschlimmern
- Legacy-Systemen könnte die architektonische Flexibilität fehlen, um saubere Degradationsgrenzen zu unterstützen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine auf einem Legacy-Monolithen gebaute E-Commerce-Plattform erlebte während saisonaler Verkehrsspitzen vollständige Ausfälle, weil ihre Empfehlungs-Engine übermäßige Datenbankressourcen verbrauchte. Das Team implementierte Graceful Degradation, indem es zwischengespeicherte, nicht personalisierte Empfehlungen auslieferte, wenn Datenbankantwortzeiten einen Schwellenwert überschritten, und Empfehlungen unter extremer Last vollständig deaktivierte. Dies hielt den Kern-Einkaufs- und Checkout-Flow während Spitzenzeiten verfügbar und verwandelte, was vollständige Ausfälle gewesen wären, in geringfügige Feature-Reduzierungen, die die meisten Kunden nie bemerkten.
