---
title: Disaster Recovery
description: Methoden zur Wiederherstellung des Betriebs nach Katastrophen oder größeren
  Störungen.
category:
- Operations
problems:
- system-outages
- single-points-of-failure
- missing-rollback-strategy
- poor-operational-concept
- monitoring-gaps
- deployment-risk
layout: solution
lang: de
en_slug: disaster-recovery
related_solutions:
- slug: regular-backups
  similarity: 0.8
- slug: restore-points
  similarity: 0.8
- slug: backup-and-recovery
  similarity: 0.8
- slug: chaos-engineering
  similarity: 0.75
- slug: incident-management
  similarity: 0.75
- slug: failover-mechanisms
  similarity: 0.75
---

## Description

Disaster Recovery ist die Menge dokumentierter Prozeduren, Infrastruktur und geübter Praktiken, die einer Organisation erlauben, ein System nach einer größeren Störung — Hardwareausfall, Standortverlust, Datenkorruption oder katastrophaler Ausfall — innerhalb eines vereinbarten Zeitrahmens und mit einer vereinbarten maximalen Menge verlorener Daten wieder in Betrieb zu setzen. Es beruht auf zwei expliziten Zielen, dem Recovery Time Objective und dem Recovery Point Objective, die die Organisation zwingen, im Voraus anzugeben, wie viel Ausfallzeit und Datenverlust sie für ein gegebenes System tatsächlich tolerieren kann, statt die Antwort während eines Ausfalls zu entdecken. Legacy-Systeme sind unverhältnismäßig stark Katastrophenszenarien ausgesetzt, weil sie häufig auf alternder, unterredundanter Infrastruktur laufen, von undokumentierten Konfigurationen abhängen und gebaut wurden, bevor formelle Kontinuitätsplanung Standardpraxis war — die Abhängigkeiten, die Wiederherstellung korrekt sequenzieren muss, sind oft nur informell bekannt, wenn überhaupt. Disaster-Recovery-Planung zwingt dieses stillschweigende Wissen ans Licht: Runbooks zu bauen und Wiederherstellung zu testen bedeutet, dass jemand zuerst etablieren muss, was tatsächlich von was abhängt, was wertvolle Grundlagenarbeit für Modernisierung ganz abgesehen von ihrer Nutzung in einem echten Notfall ist. Weil Backups und Prozeduren still verfallen, wenn sie nie geübt werden, liefert die Disziplin ihr Versprechen nur, wenn Wiederherstellung regelmäßig geübt statt als funktionierend angenommen wird.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Definieren Sie Recovery Time Objectives (RTO) und Recovery Point Objectives (RPO) für jedes kritische System basierend auf einer Geschäftsauswirkungsanalyse
- Implementieren Sie automatisierte Backups mit regelmäßiger Verifikation, dass Backups tatsächlich wiederhergestellt werden können
- Richten Sie Off-Site- oder regionsübergreifende Backup-Speicherung ein, um vor standortweiten Ausfällen zu schützen
- Erstellen Sie dokumentierte Runbooks für jedes Katastrophenszenario, die Schritt-für-Schritt-Wiederherstellungsprozeduren abdecken
- Führen Sie regelmäßige Disaster-Recovery-Übungen durch, um zu validieren, dass Prozeduren funktionieren und Teams geschult sind
- Implementieren Sie Überwachung, die Katastrophenbedingungen erkennt und wo möglich automatisierte Wiederherstellung auslöst
- Pflegen Sie ein aktuelles Inventar aller Systemabhängigkeiten, damit Wiederherstellung korrekt sequenziert werden kann

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Stellt Geschäftskontinuität während größerer Ausfälle oder katastrophaler Fehler sicher
- Reduziert die finanzielle Auswirkung von Ausfallzeit durch schnellere Wiederherstellung
- Bietet Stakeholdern Vertrauen, dass kritische Systeme wiederhergestellt werden können
- Erfüllt regulatorische und Compliance-Anforderungen für Geschäftskontinuität
- Deckt Systemabhängigkeiten und Single Points of Failure durch Planungsübungen auf

**Kosten und Risiken:**
- Die Pflege von Disaster-Recovery-Infrastruktur verdoppelt manche Infrastrukturkosten
- DR-Übungen verbrauchen Teamzeit und können den normalen Betrieb vorübergehend stören
- Ungetestete Disaster-Recovery-Pläne bieten falsches Vertrauen und können bei Bedarf versagen
- Legacy-Systeme mit undokumentierten Abhängigkeiten sind besonders schwer für DR zu planen
- Die Synchronisation von DR-Umgebungen mit Produktion erfordert laufenden Aufwand

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Das Legacy-Lagerverwaltungssystem eines Logistikunternehmens lief auf einem einzelnen physischen Server mit nächtlichen Bandsicherungen, die nie auf Wiederherstellbarkeit getestet worden waren. Als ein Storage-Controller-Ausfall das System offline nahm, entdeckte das Team, dass das jüngste wiederherstellbare Backup wegen still fehlschlagender Backup-Jobs drei Wochen alt war. Nach diesem Vorfall investierte das Unternehmen in eine automatisierte DR-Strategie: tägliche verifizierte Backups mit Wiederherstellungstests, einen Warm-Standby-Server, synchronisiert über Datenbankreplikation, und dokumentierte Runbooks für jedes Ausfallszenario. Vierteljährliche DR-Übungen deckten mehrere Wiederherstellungslücken auf und behoben sie. Als ein nachfolgender Hardwareausfall eintrat, wurde das System innerhalb von 45 Minuten mit weniger als fünf Minuten Datenverlust auf den Standby wiederhergestellt.
