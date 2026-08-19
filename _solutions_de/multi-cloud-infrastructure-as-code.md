---
title: Multi-Cloud Infrastructure as Code
description: Deklarative Bereitstellung von Infrastruktur mit anbieterunabhängigen
  Modulen für mehrere Clouds.
category:
- Operations
- Architecture
problems:
- vendor-lock-in
- vendor-dependency
- vendor-dependency-entrapment
- technology-lock-in
- configuration-drift
- complex-deployment-process
- deployment-environment-inconsistencies
- manual-deployment-processes
layout: solution
lang: de
en_slug: multi-cloud-iac
related_solutions:
- slug: infrastructure-as-code
  similarity: 0.7
- slug: immutable-infrastructure
  similarity: 0.7
- slug: containerization
  similarity: 0.65
- slug: cloud-native-development
  similarity: 0.65
- slug: virtual-networks
  similarity: 0.65
- slug: standardized-deployment-scripts
  similarity: 0.65
---

## Description

Multi-Cloud Infrastructure as Code stellt Infrastruktur deklarativ über anbieterunabhängige Module bereit — typischerweise gebaut mit Werkzeugen wie Terraform oder Pulumi —, die eine einheitliche Schnittstelle bieten, während sie die anbieterspezifischen Ressourcendefinitionen darunter abstrahieren, sodass dasselbe Modul AWS, Azure oder eine andere Cloud anvisieren kann, nur mit geänderten Variablen. Dies funktioniert, indem zuvor manuelle Konsolenkonfiguration und Ad-hoc-Shell-Skripte in versionskontrollierte, überprüfbare Definitionen kodifiziert werden, beginnend mit der einfachsten Umgebung und der Validierung, dass sie das bestehende manuelle Setup reproduziert, bevor der Ansatz weiter ausgedehnt wird. Legacy-Systeme sind häufig an einen einzigen Cloud-Anbieter gebunden, nicht weil dieser Anbieter bewusst aus technischen Gründen gewählt wurde, sondern weil die Infrastruktur über Jahre schrittweise durch manuelle Klicks und anbieterspezifische Skripte aufgebaut wurde, wobei niemand die resultierende Topologie in einer Form dokumentierte, die anderswo reproduziert werden könnte. Diese Art zufälligen Lock-ins lässt einer Organisation keinen Verhandlungsspielraum bei Anbieterpreisen und keine praktische Disaster-Recovery-Option, falls der primäre Anbieter einen Ausfall oder einen Vertragsstreit hat, da das erneute Deployment des Systems anderswo bedeuten würde, seine Infrastruktur von Grund auf neu aufzubauen. Anbieterunabhängige IaC einzuführen verwandelt Anbieter-Lock-in von einer unvermeidlichen strukturellen Bedingung in eine bewusste, überarbeitbare Wahl, obwohl die Abstraktion, die diese Portabilität ermöglicht, auch bedeutet, auf manche anbieterspezifische Optimierungen und fortgeschrittene verwaltete Dienste zu verzichten, die auf anderen Clouds kein Äquivalent haben.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Prüfen Sie bestehende Infrastrukturbereitstellungsskripte und manuelle Runbooks, um die aktuelle Deployment-Topologie zu verstehen
- Wählen Sie ein anbieterunabhängiges IaC-Werkzeug wie Terraform oder Pulumi, das mehrere Cloud-Anbieter unterstützt
- Abstrahieren Sie anbieterspezifische Ressourcendefinitionen in wiederverwendbare Module, die eine einheitliche Schnittstelle bieten
- Beginnen Sie damit, die einfachste Umgebung (z. B. Staging) zu kodifizieren, und validieren Sie Parität mit dem bestehenden manuellen Setup
- Nutzen Sie Variablen und Workspaces, um cloud-spezifische Details zu parametrisieren, während die Modulstruktur identisch bleibt
- Integrieren Sie IaC in CI/CD-Pipelines, damit Infrastrukturänderungen Code-Review und automatisierte Validierung durchlaufen
- Pflegen Sie eine State-Management-Strategie mit Remote-Backends und State-Locking, um Drift zu verhindern

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verringert Anbieter-Lock-in, indem Cloud-Anbieterwechsel zu einer Konfigurationsänderung statt einer Neufassung werden
- Sichert Umgebungskonsistenz durch deklarative, versionskontrollierte Infrastrukturdefinitionen
- Beseitigt manuelle Bereitstellungsfehler und Konfigurationsdrift über Umgebungen hinweg
- Ermöglicht Disaster-Recovery-Szenarien, in denen Workloads auf einer alternativen Cloud erneut deployt werden können

**Kosten und Risiken:**
- Anbieterunabhängige Abstraktionen könnten cloud-spezifische Optimierungen und fortgeschrittene Features opfern
- Die Pflege von Multi-Cloud-Modulen fügt im Vergleich zu Einzelanbieter-Vorlagen Komplexität hinzu
- State-Management über Anbieter hinweg führt zusätzliche betriebliche Last ein
- Teams brauchen Schulung in IaC-Tooling und cloud-agnostischen Designmustern
- Nicht alle Dienste haben äquivalente Angebote über Cloud-Anbieter hinweg

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Gesundheits-SaaS-Anbieter war an einen einzigen Cloud-Anbieter gebunden, dessen Preise über zwei Jahre um 40 % gestiegen waren. Ihre Infrastruktur war durch eine Mischung aus Konsolenklicks und Shell-Skripten bereitgestellt, was eine Migration unmöglich erscheinen ließ. Das Team übernahm Terraform mit anbieterunabhängigen Modulen, beginnend mit der Kodifizierung ihrer Staging-Umgebung. Über sechs Monate erstellten sie Module für Rechenleistung, Netzwerk, Speicher und Datenbankressourcen, die entweder AWS oder Azure anvisieren konnten. Als sich die Vertragsneuverhandlung festfuhr, demonstrierten sie die Fähigkeit, ihren vollständigen Stack innerhalb von Stunden auf der alternativen Cloud bereitzustellen, was ihnen erheblichen Verhandlungsspielraum gab und letztlich zu besseren Preiskonditionen führte.
