---
title: Virtualisierung
description: Isolierung von Anwendungen mit eigener
  Betriebssysteminstanz, um Ressourcen- und Abhängigkeitskonflikte zu
  vermeiden.
category:
- Operations
- Architecture
problems:
- deployment-environment-inconsistencies
- dependency-version-conflicts
- shared-dependencies
- configuration-drift
- poor-system-environment
- resource-contention
- technology-lock-in
layout: solution
lang: de
en_slug: virtualization
related_solutions:
- slug: containerization
  similarity: 0.7
- slug: virtual-networks
  similarity: 0.7
- slug: virtual-development-environments
  similarity: 0.65
- slug: emulation
  similarity: 0.65
- slug: immutable-infrastructure
  similarity: 0.65
- slug: cloud-native-development
  similarity: 0.65
---

## Description

Virtualisierung gibt einer Anwendung ihre eigene Betriebssysteminstanz, isoliert von allem anderen, was auf der zugrunde liegenden physischen Hardware läuft, sodass ihre spezifischen Laufzeit-, Bibliotheks- und Konfigurationsanforderungen nicht mehr mit denen einer anderen Anwendung, die dieselbe Maschine teilt, koexistieren müssen — und potenziell mit ihnen in Konflikt geraten. Dies löst direkt eine häufige Legacy-Pathologie: mehrere Anwendungen, die sich über Jahre auf demselben Bare-Metal-Server angesammelt haben, jede abhängig von einer anderen, manchmal inkompatiblen Version einer gemeinsam genutzten Laufzeit oder Bibliothek, sodass das Patchen oder Aktualisieren einer Anwendung riskiert, eine andere still zu brechen, die zufällig denselben Host teilt. Indem jeder Legacy-Anwendung ihr eigenes VM-Image gegeben wird, das die exakten OS-, Laufzeit- und Abhängigkeitsversionen erfasst, die sie braucht, lässt Virtualisierung Anwendungen mit widersprüchlichen oder sogar sich gegenseitig ausschließenden Anforderungen sicher auf derselben physischen Infrastruktur koexistieren, und Infrastructure-as-Code-Tooling macht diese Umgebung über Entwicklung, Staging und Produktion hinweg reproduzierbar, statt still zwischen ihnen abzudriften. Die Snapshot-Fähigkeit gibt Teams auch die Zuversicht, riskante Änderungen an brüchigen Legacy-Systemen zu versuchen, da ein schlechter Patch oder Upgrade innerhalb von Minuten auf einen bekannt-guten Zustand zurückgerollt werden kann, statt eine langwierige manuelle Wiederherstellung zu erfordern. Die Kosten sind der Overhead, ein vollständiges OS pro Instanz zu betreiben, und die betriebliche Fähigkeit, die zur Verwaltung einer Virtualisierungsplattform benötigt wird, weshalb leichtgewichtigere Containerisierung oft bevorzugt wird, wo die OS-Ebenen-Anforderungen einer Legacy-Anwendung es erlauben.

## How to Apply ◆

- Migrieren Sie Legacy-Anwendungen von Bare-Metal-Shared-Servern zu individuellen virtuellen Maschinen, wobei jede Anwendung ihr eigenes OS und ihren eigenen Abhängigkeits-Stack erhält.
- Nutzen Sie Infrastructure-as-Code-Werkzeuge (Terraform, Ansible), um virtuelle Umgebungen reproduzierbar zu definieren und bereitzustellen.
- Erstellen Sie VM-Images, die das exakte OS, die Laufzeit und die Bibliotheksversionen erfassen, die eine Legacy-Anwendung benötigt.
- Nutzen Sie Snapshots für sicheren Rollback beim Anwenden von Patches oder Konfigurationsänderungen an Legacy-Systemen.
- Konsolidieren Sie unterausgelastete physische Server durch Virtualisierung, um Hardwarekosten zu reduzieren, während Isolation erhalten bleibt.
- Erwägen Sie Containerisierung (Docker) für leichtgewichtigere Isolation, wo die OS-Anforderungen der Legacy-Anwendung dies erlauben.

## Tradeoffs ⇄

**Vorteile:**
- Beseitigt Abhängigkeitskonflikte zwischen Anwendungen, die unterschiedliche Bibliotheks- oder Laufzeitversionen benötigen.
- Ermöglicht konsistente Umgebungsreproduktion über Entwicklung, Staging und Produktion hinweg.
- Bietet Isolation, sodass der Ressourcenverbrauch einer Anwendung andere nicht beeinträchtigt.
- Vereinfacht die Notfallwiederherstellung durch VM-Snapshots und imagebasierte Backups.

**Kosten:**
- Fügt Overhead für die Verwaltung der Virtualisierungsinfrastruktur hinzu (Hypervisor, Image-Speicherung, Networking).
- VMs verbrauchen mehr Ressourcen als Container aufgrund des vollständigen OS-Overheads pro Instanz.
- Legacy-Anwendungen mit hardwarespezifischen Abhängigkeiten lassen sich möglicherweise nicht sauber virtualisieren.
- Erfordert betriebliche Fähigkeiten in Virtualisierungsplattformen, die Teams möglicherweise erwerben müssen.
- Lizenzkosten für Betriebssysteme und Virtualisierungsplattformen können erheblich sein.

## How It Could Be

Eine Regierungsbehörde betreibt mehrere Legacy-Anwendungen auf gemeinsam genutzten Windows-Servern, wo widersprüchliche .NET-Framework-Versionen und DLL-Abhängigkeiten häufige Deployment-Fehlschläge verursachen. Durch die Virtualisierung jeder Anwendung in ihre eigene VM mit einem festen OS-Image werden Abhängigkeitskonflikte beseitigt. Das Infrastrukturteam nutzt Ansible, um VMs aus versionierten Vorlagen bereitzustellen, was sicherstellt, dass Entwicklungsumgebungen exakt der Produktion entsprechen. Als eine kritische Legacy-Anwendung eine ältere Laufzeit braucht, die mit Sicherheitspatches kollidiert, die eine andere Anwendung benötigt, erlaubt die von der Virtualisierung gebotene Isolation, dass beide ohne Kompromiss koexistieren. Die VM-Snapshot-Fähigkeit gibt dem Team auch die Zuversicht, Upgrades zu versuchen, in dem Wissen, dass sie innerhalb von Minuten zurückgerollt werden können.
