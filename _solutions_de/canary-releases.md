---
title: Canary Releases
description: Schrittweise Einführung von Änderungen für eine begrenzte Nutzergruppe,
  um Risiko zu minimieren.
category:
- Operations
- Process
problems:
- deployment-risk
- large-risky-releases
- frequent-hotfixes-and-rollbacks
- release-instability
- release-anxiety
- fear-of-change
- high-defect-rate-in-production
layout: solution
lang: de
en_slug: canary-releases
related_solutions:
- slug: rollback-mechanisms
  similarity: 0.8
- slug: dark-launches
  similarity: 0.8
- slug: chaos-engineering
  similarity: 0.8
- slug: continuous-integration-and-delivery
  similarity: 0.8
- slug: rolling-updates
  similarity: 0.75
- slug: error-budgets
  similarity: 0.75
---

## Description

Ein Canary Release deployt eine neue Softwareversion an eine kleine, kontrollierte Untermenge des Produktions-Traffics, während die Mehrheit weiterhin auf der bewährten stabilen Version läuft, unter Nutzung von Traffic-Routing (gewichtetes Load Balancing, Feature Flags oder Service-Mesh-Regeln), um die Exposition zu kontrollieren. Die Technik ersetzt einen einzelnen folgenreichen Umstieg durch einen abgestuften, beobachtbaren Rollout: Gesundheitsmetriken der Canary-Population werden gegen die Baseline verglichen, und das Release wird entweder schrittweise befördert oder automatisch zurückgesetzt, wenn es sich verschlechtert. Für Legacy-Systeme, bei denen Releases oft selten, schlecht getestet und mit hoher Angst verbunden sind, weil ein vollständiges Versagen kostspielig und schwer umkehrbar ist, verwandeln Canary Releases den Moment des Deployments von einem binären Glücksspiel in eine Reihe kleiner, umkehrbarer Wetten. Weil nur ein Bruchteil der Nutzer zu einem beliebigen Zeitpunkt exponiert ist, werden Defekte, die sonst organisationsweite Ausfälle verursachen würden, eingedämmt und früh abgefangen, unter Nutzung echter Produktionsbedingungen statt Staging-Annäherungen. Dies ist besonders wertvoll, wenn Legacy-Test-Suiten dünn oder unzuverlässig sind, da Canary Releases eine Schicht empirischer, metrikbasierter Validierung hinzufügen, die nicht davon abhängt, dass Tests korrekt jeden Fehlermodus antizipieren. Der Ansatz setzt Infrastruktur voraus, die in der Lage ist, zwei Versionen gleichzeitig laufen zu lassen und Traffic zwischen ihnen zu routen, was oft die schwierigere Legacy-Modernisierungsvoraussetzung ist, die zuerst gebaut werden muss.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Richten Sie Infrastruktur zum Routing eines konfigurierbaren Prozentsatzes des Traffics zur neuen Version neben der stabilen Version ein
- Definieren Sie Schlüssel-Gesundheitsmetriken (Fehlerrate, Latenz, Geschäftskennzahlen), die während der Canary-Phase überwacht werden
- Beginnen Sie mit einem kleinen Prozentsatz (1-5 %) des Traffics, der zum Canary geleitet wird, und erhöhen Sie schrittweise basierend auf Metrikschwellen
- Implementieren Sie automatisierte Rollback-Auslöser, die Traffic zur stabilen Version zurücksetzen, wenn sich Gesundheitsmetriken verschlechtern
- Nutzen Sie Feature Flags in Kombination mit Canary-Routing, um zu kontrollieren, welche Features für Canary-Nutzer aktiv sind
- Etablieren Sie ein minimales Beobachtungsfenster vor jeder Traffic-Erhöhung, um sich langsam entwickelnde Probleme zu erkennen
- Stellen Sie sicher, dass Logging und Monitoring zwischen Canary- und stabilem Traffic unterscheiden, für genauen Vergleich

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Begrenzt den Explosionsradius fehlerhafter Releases auf eine kleine Untermenge von Nutzern
- Bietet echte Produktionsvalidierung vor vollständigem Rollout
- Ermöglicht datengetriebene Rollout-Entscheidungen basierend auf tatsächlichen Metriken statt Annahmen
- Verringert den Druck und die Angst im Zusammenhang mit Big-Bang-Releases
- Erlaubt schnellen Rollback durch einfaches Umleiten des Traffics vom Canary weg

**Kosten und Risiken:**
- Erfordert Infrastruktur, die zwei Versionen gleichzeitig laufen lassen und Traffic aufteilen kann
- Datenbankschemaänderungen müssen abwärtskompatibel sein, um beide Versionen gleichzeitig zu unterstützen
- Monitoring- und Metrikvergleichsinfrastruktur muss vorhanden sein, bevor Canary Releases nützlich sind
- Kleine Canary-Populationen könnten Probleme nicht aufdecken, die nur im großen Maßstab auftreten
- Fügt dem Deployment-Prozess operative Komplexität hinzu

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Reisebuchungsplattform hatte eine Geschichte disruptiver Produktionsvorfälle nach Releases, was das Team dazu brachte, nur einmal pro Quartal mit umfangreichem manuellem Testen zu releasen. Das Team implementierte Canary Releases unter Nutzung der gewichteten Routing-Fähigkeit ihres Load Balancers. Neue Versionen wurden anfänglich 2 % des Traffics exponiert, mit automatisierten Gesundheitsprüfungen, die Buchungsabschlussraten und Fehlerraten überwachten. Wenn Metriken zwei Stunden stabil blieben, wurde der Traffic schrittweise auf 10 %, 50 % und dann 100 % erhöht. Dieser Ansatz fing einen kritischen Zahlungsintegrationsbug während einer Canary-Phase ab, der nur 2 % der Nutzer statt der gesamten Kundenbasis betraf. Die Release-Häufigkeit stieg von vierteljährlich auf wöchentlich.
