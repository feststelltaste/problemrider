---
title: Connection Pooling
description: Wiederverwendung bereits etablierter Verbindungen statt Neuerstellung
  pro Anfrage.
category:
- Performance
- Database
problems:
- database-connection-leaks
- misconfigured-connection-pools
- high-connection-count
- slow-application-performance
- incorrect-max-connection-pool-size
- high-database-resource-utilization
- unreleased-resources
layout: solution
lang: de
en_slug: connection-pooling
related_solutions:
- slug: distributed-caching
  similarity: 0.8
- slug: resource-pooling
  similarity: 0.8
- slug: lazy-loading
  similarity: 0.8
- slug: reactive-programming
  similarity: 0.75
- slug: rate-limiting
  similarity: 0.75
- slug: parallelization
  similarity: 0.75
---

## Description

Connection Pooling unterhält eine Menge vorab etablierter, wiederverwendbarer Verbindungen zu einer Ressource — am häufigsten eine Datenbank, aber gleichermaßen anwendbar auf HTTP-Clients, LDAP-Server oder Message Broker —, sodass Anfragen eine bereite Verbindung aus dem Pool ausleihen, statt die Kosten zu zahlen, jedes Mal eine neue zu etablieren und abzubauen. Die Etablierung einer Verbindung beinhaltet typischerweise einen TCP-Handshake, Authentifizierung und oft TLS-Aushandlung, alles feste Kosten, die mit dem Anfragevolumen statt mit tatsächlich geleisteter Arbeit skalieren; unter Last kann dieser Overhead allein eine Datenbank an ihr Verbindungslimit drängen, lange bevor ihr die echte Kapazität ausgeht. Legacy-Anwendungen erstellen häufig aus Einfachheitsgründen eine Verbindung pro Anfrage, ein Muster, das bei geringem Traffic unsichtbar war und erst zu einem Engpass wird, während die Nutzung wächst, oft verschärft durch Verbindungslecks, bei denen Code eine Verbindung erwirbt, sie aber nie zurückgibt. Pooling begrenzt die Anzahl gleichzeitiger Verbindungen auf eine Größe, die die dahinterliegende Ressource tatsächlich tragen kann, was sowohl Antwortzeiten verbessert, indem Setup-Latenz vom Anfragepfad entfernt wird, als auch die Ressource davor schützt, durch unbegrenztes Verbindungswachstum überwältigt zu werden. Die Pool-Größe, das Timeout und die Validierungseinstellungen richtig zu setzen ist essenziell, da ein zu klein dimensionierter oder falsch konfigurierter Pool den Engpass lediglich von der Verbindungserstellung zur Verbindungswarteschlange verlagert.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Ersetzen Sie direkte Verbindungserstellung durch eine für den Technologie-Stack angemessene Connection-Pool-Bibliothek (HikariCP, pgBouncer, c3p0)
- Dimensionieren Sie den Pool basierend auf tatsächlichen nebenläufigen Nutzungsmustern, nicht willkürlichen großen Zahlen
- Konfigurieren Sie angemessene Verbindungsvalidierungs- und Eviction-Richtlinien, um veraltete oder defekte Verbindungen zu handhaben
- Setzen Sie Verbindungstimeouts und maximale Wartezeiten, sodass die Anwendung schnell scheitert statt zu hängen
- Überwachen Sie Pool-Metriken: aktive Verbindungen, untätige Verbindungen, Wartezeiten und Verbindungserstellungsraten
- Auditieren Sie Legacy-Code auf Verbindungslecks, wo Verbindungen erworben, aber nicht ordentlich an den Pool zurückgegeben werden
- Wenden Sie Connection Pooling auf alle externen Ressourcen an: Datenbanken, HTTP-Clients, LDAP-Verbindungen, Message Broker

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Eliminiert den Overhead der Etablierung neuer Verbindungen für jede Anfrage (TCP-Handshake, Authentifizierung, SSL-Aushandlung)
- Bietet vorhersehbaren Ressourcenverbrauch durch Begrenzung der maximalen Verbindungsanzahl
- Verbessert Antwortzeiten durch verfügbare, vorab etablierte, gebrauchsfertige Verbindungen
- Verringert die Last auf dem Datenbankserver durch Begrenzung gleichzeitiger Verbindungen

**Kosten und Risiken:**
- Falsch dimensionierte Pools können Verbindungshunger (zu klein) oder Ressourcenverschwendung (zu groß) verursachen
- Veraltete Verbindungen im Pool können intermittierende Fehler verursachen, wenn Validierung nicht konfiguriert ist
- Connection Pools fügen Konfigurationskomplexität hinzu, die für die spezifische Arbeitslast abgestimmt werden muss
- Pool-Erschöpfung unter Last kann kaskadierende Ausfälle verursachen, wenn nicht mit angemessenen Timeouts gehandhabt

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-Java-Webanwendung erstellte für jede HTTP-Anfrage eine neue Datenbankverbindung und schloss sie am Ende des Anfrage-Handlers. Unter Last erreichte der Datenbankserver sein maximales Verbindungslimit, was neue Anfragen mit Connection-Refused-Fehlern scheitern ließ. Das Team führte HikariCP mit einem Pool von 20 Verbindungen ein, abgestimmt auf das empfohlene Maximum der Datenbank für die Anwendung. Der Verbindungsetablierungs-Overhead verschwand vom Anfragepfad, die durchschnittlichen Antwortzeiten verbesserten sich um 15 %, und die CPU-Auslastung des Datenbankservers sank, weil er keine Zyklen mehr für die Verwaltung Tausender kurzlebiger Verbindungen pro Minute aufwendete.
