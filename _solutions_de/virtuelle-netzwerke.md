---
title: Virtuelle Netzwerke
description: Abstraktion von Netzwerkkonfigurationen durch virtuelle
  Netzwerke.
category:
- Operations
- Architecture
problems:
- technology-lock-in
- vendor-lock-in
- deployment-environment-inconsistencies
- configuration-drift
- poor-system-environment
- network-latency
layout: solution
lang: de
en_slug: virtual-networks
related_solutions:
- slug: containerization
  similarity: 0.75
- slug: protocol-abstraction
  similarity: 0.75
- slug: database-abstraction
  similarity: 0.75
- slug: abstraction-layers
  similarity: 0.75
- slug: virtual-development-environments
  similarity: 0.75
- slug: platform-independent-configuration-management
  similarity: 0.75
---

## Description

Ein virtuelles Netzwerk abstrahiert die Netzwerkkonfiguration einer Anwendung von jeder spezifischen physischen Topologie — Hardware-Appliances, festen IP-Bereichen, VLANs — weg, indem Konnektivität und Richtlinie als Software ausgedrückt werden, sei es durch Cloud Virtual Private Clouds, Overlay-Netzwerke oder Software-Defined-Networking-Werkzeuge, und indem hartcodierte Adressen durch DNS-basierte Service Discovery ersetzt werden. Legacy-Systeme sammeln häufig eine tiefe, oft unsichtbare Abhängigkeit vom spezifischen physischen Netzwerk an, in das sie ursprünglich eingesetzt wurden: Konfigurationsdateien kodieren IP-Adressen hart, Anwendungslogik nimmt eine bestimmte VLAN-Segmentierung an, und niemand im aktuellen Team kann mit Zuversicht sagen, welche dieser Annahmen tragend versus beiläufig sind. Diese Kopplung wird zu einem direkten Blocker für Infrastrukturänderungen — eine Rechenzentrumskonsolidierung oder eine Cloud-Migration kann nicht fortschreiten, bis jede dieser impliziten Topologieannahmen gefunden und adressiert wurde, und sie allein durch Inspektion zu finden ist langsam und fehleranfällig. Die Einführung einer virtuellen Netzwerkschicht lässt die Anwendung weiterhin die Dienste, von denen sie abhängt, über stabile, abstrakte Namen statt physische Adressen auflösen, was die Anwendung davon entkoppelt, wo diese Dienste tatsächlich zu einem gegebenen Zeitpunkt zufällig laufen. Dies macht es möglich, die zugrunde liegende Infrastruktur zu verschieben — zwischen Rechenzentren, in die Cloud, oder zwischen Umgebungen —, ohne den Anwendungscode anzufassen, der davon abhängt, auf Kosten einer zusätzlichen Abstraktionsschicht, die Low-Level-Netzwerk-Troubleshooting aufwändiger machen kann als die Arbeit mit einer direkten physischen Verbindung.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Kartieren Sie die bestehende physische Netzwerktopologie und identifizieren Sie Abhängigkeiten von spezifischen IP-Bereichen, VLANs oder Hardware-Appliances
- Führen Sie Software-Defined Networking (SDN) oder Cloud Virtual Private Clouds ein, um die Anwendung von der physischen Netzwerkinfrastruktur zu entkoppeln
- Nutzen Sie Overlay-Netzwerke (z. B. Docker-Netzwerke, Kubernetes-Netzwerkrichtlinien, VXLAN), um portable Netzwerkkonfigurationen zu schaffen
- Ersetzen Sie hartcodierte IP-Adressen durch DNS-basierte Service Discovery, sodass Anwendungen Endpunkte dynamisch auflösen
- Definieren Sie Netzwerkrichtlinien als Code mit Werkzeugen wie Terraform oder Calico, um Konsistenz über Umgebungen hinweg sicherzustellen
- Testen Sie Netzwerkkonfigurationen in isolierten virtuellen Umgebungen, bevor Sie sie auf Produktion anwenden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Ermöglicht konsistente Netzwerkkonfiguration über Entwicklungs-, Staging- und Produktionsumgebungen hinweg
- Vereinfacht die Migration zwischen Rechenzentren oder Cloud-Anbietern, da die Netzwerktopologie abstrahiert ist
- Erlaubt schnelle Bereitstellung isolierter Testumgebungen mit produktionsähnlichem Networking
- Reduziert die Abhängigkeit von physischer Netzwerkhardware und spezifischen Anbieterkonfigurationen

**Kosten und Risiken:**
- Virtuelle Netzwerk-Overlays fügen Latenz und Komplexität im Vergleich zu direktem physischem Networking hinzu
- Die Fehlersuche bei Netzwerkproblemen wird mit zusätzlichen Abstraktionsschichten schwieriger
- Teams brauchen neue Fähigkeiten in Software-Defined Networking und Cloud-Networking-Konzepten
- Manche Legacy-Anwendungen nehmen spezifische Netzwerktopologien an, die schwer zu virtualisieren sind

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Gesundheitsunternehmen betrieb ein Legacy-System, verteilt über drei physische Rechenzentren, mit Netzwerkkonfigurationen, die manuell von einem kleinen Infrastrukturteam verwaltet wurden. Der Umzug in die Cloud war blockiert, weil die Anwendung sich auf spezifische IP-Adressbereiche und VLAN-Konfigurationen verließ, die in Dutzenden von Konfigurationsdateien hartcodiert waren. Das Team führte Kubernetes mit Calico-Netzwerkrichtlinien ein und ersetzte IP-basierte Adressierung durch DNS-Service-Discovery. Netzwerkrichtlinien wurden als Code definiert und konsistent über alle Umgebungen angewendet. Die Migration zur Cloud-Infrastruktur wurde abgeschlossen, ohne Anwendungscode zu ändern, und neue Testumgebungen konnten mit identischen Netzwerkkonfigurationen in Minuten statt Wochen bereitgestellt werden.
