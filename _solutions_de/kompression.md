---
title: Kompression
description: Reduzierung des Speicherbedarfs mit oder ohne Verlust.
category:
- Performance
problems:
- slow-application-performance
- network-latency
- excessive-disk-io
- unbounded-data-growth
- high-client-side-resource-consumption
- unoptimized-file-access
- serialization-deserialization-bottlenecks
layout: solution
lang: de
en_slug: compression
related_solutions:
- slug: image-and-asset-optimization
  similarity: 0.75
- slug: distributed-caching
  similarity: 0.75
- slug: data-archiving
  similarity: 0.75
- slug: data-deduplication
  similarity: 0.75
- slug: lazy-loading
  similarity: 0.75
- slug: connection-pooling
  similarity: 0.75
---

## Description

Kompression kodiert Daten in eine kleinere Repräsentation um, entweder verlustfrei (sodass das Original exakt rekonstruiert werden kann, wie bei gzip oder Brotli) oder verlustbehaftet (eine gewisse Qualitätsverringerung im Austausch für eine weit kleinere Größe akzeptierend, wie bei bestimmten Bild- oder Audioformaten), und sie kann auf Protokollebene, im Ruhezustand oder innerhalb von Anwendungs-Payloads angewendet werden. Legacy-Systeme speichern und übertragen Daten häufig in ausführlichen, unkomprimierten Formen, weil Kompression nach der ursprünglichen Implementierung nie überarbeitet wurde, was große XML- oder Text-Payloads, stetig wachsende Log-Dateien und Archivdaten hinterlässt, die Festplattenspeicher und Netzwerkbandbreite weit über das hinaus verbrauchen, was der tatsächliche Informationsgehalt erfordert. Kompression an der richtigen Schicht anzuwenden — HTTP-Kompression für API-Antworten, Ruhezustandskompression für selten zugegriffene Archivdaten, Protokollebenen-Kompression für Inter-Service-Traffic — greift direkt Speicherwachstum und Übertragungslatenz an, ohne irgendeine Änderung am zugrunde liegenden Datenmodell oder der Geschäftslogik zu erfordern. Die Wahl des Algorithmus ist wichtig: Schnelle, niedrigratige Algorithmen passen zu Echtzeitpfaden, wo CPU-Overhead minimal bleiben muss, während hochratige Algorithmen für Archivdaten angemessen sind, die selten gelesen werden. Weil Kompression CPU-Zyklen gegen Speicher und Bandbreite eintauscht, ist sie besonders effektiv als risikoarme, inkrementelle Intervention in Legacy-Systemen, wo sich der Engpass über die Jahre von Rechenleistung zu E/A- oder Netzwerkübertragung verschoben hat.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Aktivieren Sie HTTP-Kompression (gzip, Brotli) auf Webservern und API-Gateways für textbasierte Antworten
- Komprimieren Sie Daten im Ruhezustand in Datenbanken und Dateispeicher für Archiv- und selten zugegriffene Daten
- Nutzen Sie Protokollebenen-Kompression für Inter-Service-Kommunikation (gRPC, komprimierte Nachrichtenwarteschlangen-Payloads)
- Wählen Sie Kompressionsalgorithmen angemessen für den Datentyp und das Zugriffsmuster: schnelle Algorithmen für Echtzeit, hochratige Algorithmen für Archiv
- Implementieren Sie Kompression für Log-Dateien und Prüfpfade, die unbegrenzt wachsen
- Testen Sie Kompressionsraten und CPU-Overhead mit repräsentativen Produktionsdaten, bevor Sie deployen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Verringert den Netzwerkbandbreitenverbrauch, was Übertragungszeiten besonders über langsame Verbindungen verbessert
- Verringert Speicherkosten für Daten im Ruhezustand
- Verbessert die Cache-Effizienz, indem mehr Daten in begrenzten Cache-Speicher passen
- Kann Seitenladezeiten für Webanwendungen erheblich verbessern

**Kosten und Risiken:**
- Kompression und Dekompression verbrauchen CPU-Zyklen, was für CPU-gebundene Systeme ein Engpass sein könnte
- Verlustbehaftete Kompression (für Bilder, Audio) verringert Datenqualität dauerhaft
- Komprimierte Daten sind ohne Dekompressionswerkzeuge schwerer zu inspizieren und zu debuggen
- Manche Datentypen (bereits komprimierte Bilder, verschlüsselte Daten) komprimieren nicht gut und verschwenden CPU

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Dokumentenmanagementsystem speicherte XML-basierte Dokumente unkomprimiert, wobei über 8 TB Speicher verbraucht wurden, die um 500 GB pro Quartal wuchsen. API-Antworten für Dokumentabruf waren aufgrund der großen Payload-Größen langsam. Das Team aktivierte gzip-Kompression am API-Gateway, was die Antwortgrößen für XML-Payloads um ungefähr 85 % verringerte. Für Speicher implementierten sie transparente Kompression auf Datenbankebene für Dokumente älter als 90 Tage. Diese Änderungen setzten sofort 5 TB Speicher frei und verringerten die Dokumentabrufzeiten von 3 Sekunden auf unter 500 Millisekunden für typische Dokumente.
