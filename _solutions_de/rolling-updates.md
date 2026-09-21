---
title: Rolling Updates
description: Schrittweise Aktualisierung von Servern oder Instanzen.
category:
- Operations
problems:
- deployment-risk
- system-outages
- large-risky-releases
- complex-deployment-process
- deployment-coupling
- release-instability
layout: solution
lang: de
en_slug: rolling-updates
related_solutions:
- slug: rollback-mechanisms
  similarity: 0.75
- slug: canary-releases
  similarity: 0.75
- slug: regular-maintenance-and-updates
  similarity: 0.75
- slug: restore-points
  similarity: 0.7
- slug: chaos-engineering
  similarity: 0.7
- slug: blue-green-canary-deployments
  similarity: 0.7
---

## Description

Rolling Updates stellen eine neue Version eines Systems inkrementell bereit, Instanz für Instanz oder in kleinen Batches, statt eine gesamte Serverflotte gleichzeitig zu ersetzen, wobei Health Checks jede aktualisierte Instanz validieren, bevor das Rollout zur nächsten fortschreitet. Da alte und neue Versionen der Anwendung während der Dauer des Rollouts nebeneinander laufen, beseitigt dieser Ansatz sowohl die Ausfallzeit, die eine Big-Bang-Bereitstellung sonst erfordern würde, als auch begrenzt er den Explosionsradius einer schlechten Veröffentlichung auf den Anteil der Flotte, der aktualisiert wurde, als ein Problem erstmals erkannt wird. Für Legacy-Systeme, die oft als feste Menge langlebiger Server statt dynamisch skalierter Infrastruktur bereitgestellt werden, bieten Rolling Updates einen Weg, Bereitstellungsrisiko zu reduzieren, ohne das System zuerst in etwas Cloud-nativeres umzuarchitektieren, da die Technik auf der Bereitstellungsebene operiert, statt Änderungen an der Anwendung selbst zu erfordern. Der Ansatz erlegt der Anwendung jedoch eine bedeutende Vorbedingung auf: Sie muss tolerieren, dass zwei Versionen gleichzeitig laufen, was bedeutet, dass Datenbankschemaänderungen und jeglicher gemeinsamer Zustand während der Dauer des Rollouts über beide Versionen hinweg kompatibel bleiben müssen — eine Einschränkung, die Legacy-Anwendungen mit eng gekoppelten Schemata oder Singleton-In-Memory-Zustand ohne zusätzliche Arbeit möglicherweise nicht erfüllen. Wo diese Vorbedingung erfüllt werden kann, verwandeln Rolling Updates das, was zuvor ein geplantes Wartungsfenster mit voller Flottenexposition war, in einen inkrementellen, selbstprüfenden Prozess, der automatisch anhält, sobald frühe Batches Anzeichen von Problemen zeigen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Konfigurieren Sie Bereitstellungs-Tooling, um Instanzen einzeln oder in kleinen Batches statt alle auf einmal zu aktualisieren
- Implementieren Sie Health Checks, die jede aktualisierte Instanz validieren, bevor zur nächsten fortgeschritten wird
- Definieren Sie automatische Rollback-Trigger, die das Rolling Update anhalten, wenn Fehlerraten Schwellwerte überschreiten
- Stellen Sie sicher, dass die Legacy-Anwendung unterstützt, alte und neue Versionen während des Übergangs gleichzeitig laufen zu lassen
- Behandeln Sie Datenbankschemaänderungen sorgfältig, um Kompatibilität mit beiden Versionen während des Aktualisierungsfensters aufrechtzuerhalten
- Nutzen Sie Connection Draining, um Instanzen elegant aus der Load-Balancer-Rotation zu entfernen, bevor Sie sie aktualisieren
- Überwachen Sie Schlüsselmetriken während des Rollouts und pausieren Sie, wenn Anomalien erkannt werden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Beseitigt Bereitstellungsausfallzeit durch Aufrechterhaltung der Diensteverfügbarkeit während der gesamten Aktualisierung
- Begrenzt den Explosionsradius, da nur ein Teil der Instanzen zu einem gegebenen Zeitpunkt die neue Version ausführt
- Ermöglicht frühe Erkennung von Problemen, bevor sie alle Instanzen betreffen
- Bietet natürliche Kontrollpunkte für automatisierte Rollback-Entscheidungen

**Kosten und Risiken:**
- Alte und neue Versionen müssen koexistieren, was rückwärtskompatible Änderungen erfordert
- Rolling Updates dauern länger als Bereitstellungen der gesamten Flotte
- Das Debuggen von Problemen während gemischter Versionszustände kann herausfordernd sein
- Legacy-Anwendungen mit gemeinsamem Zustand oder Singleton-Mustern unterstützen möglicherweise kein schrittweises Rollout

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Medien-Streaming-Unternehmen stellte seine Legacy-Content-Delivery-Anwendung über 12 Server bereit, wobei zuvor alle Server gleichzeitig während eines 30-minütigen Wartungsfensters aktualisiert wurden. Durch die Implementierung von Rolling Updates, die jeweils zwei Server gleichzeitig mit Health-Check-Validierung zwischen Batches aktualisierten, beseitigte das Team geplante Ausfallzeit vollständig. Als eine Bereitstellung ein Speicherleck einführte, wurde es während der Aktualisierung des ersten Batches durch Health-Check-Fehlschläge erkannt, und nur zwei von zwölf Servern waren betroffen. Die Bereitstellung wurde automatisch angehalten und auf diesen zwei Servern zurückgesetzt, was jegliche für Nutzer sichtbare Auswirkung verhinderte.
