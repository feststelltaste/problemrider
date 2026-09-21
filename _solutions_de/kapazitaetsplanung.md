---
title: Kapazitätsplanung
description: Schätzung künftigen Ressourcenbedarfs aus Wachstumsprognosen und Performance-Modellen.
category:
- Performance
- Operations
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/capacity-planning/
problems:
- capacity-mismatch
- scaling-inefficiencies
- growing-task-queues
- task-queues-backing-up
- work-queue-buildup
- insufficient-worker-capacity
- thread-pool-exhaustion
- high-connection-count
- incorrect-max-connection-pool-size
- load-balancing-problems
- misconfigured-connection-pools
- resource-allocation-failures
- resource-waste
- memory-swapping
- virtual-memory-thrashing
- index-fragmentation
- rate-limiting-issues
- unused-indexes
layout: solution
lang: de
en_slug: capacity-planning
related_solutions:
- slug: proactive-capacity-management
  similarity: 0.85
- slug: performance-modeling
  similarity: 0.8
- slug: resource-usage-optimization
  similarity: 0.8
- slug: monitoring-system-utilization
  similarity: 0.75
- slug: service-level-objectives
  similarity: 0.75
- slug: capacity-based-planning
  similarity: 0.75
---

## Description

Kapazitätsplanung prognostiziert künftigen Ressourcenbedarf — Datenbankverbindungen, Worker-Threads, Speicher, Warteschlangendurchsatz — durch Korrelation gemessenen aktuellen Verbrauchs mit Geschäftswachstumsprognosen, statt die Grenzen des Systems erst zu entdecken, wenn es unter Last versagt. Legacy-Systeme laufen sehr häufig ohne jegliches solches Modell: Pool-Größen und Thread-Zahlen, einmal beim initialen Deployment gesetzt, nie überprüft, während die Nutzung Jahr für Jahr wuchs, bis eine routinemäßige Lasterhöhung eine Schwelle überschreitet, auf die niemand geachtet hat. Selbst ein einfaches Modell aus einer gemessenen Baseline aufzubauen und Alarmschwellen weit vor tatsächlicher Sättigung zu setzen, verwandelt das, was sonst ein Notfall-Skalierungsvorfall wäre, in eine geplante, unaufgeregte Kapazitätsänderung.

## How to Apply ◆

> Legacy-Systeme laufen oft ohne formales Verständnis ihrer Ressourcengrenzen oder Wachstumstrajektorien. Kapazitätsplanung führt einen disziplinierten Ansatz zur Bedarfsprognose ein und stellt sicher, dass das System sie erfüllen kann, bevor Ausfälle auftreten.

- Etablieren Sie eine Baseline, indem Sie den aktuellen Ressourcenverbrauch des Legacy-Systems unter normalen und Spitzenbedingungen messen. Erfassen Sie CPU, Speicher, Festplatten-E/A, Netzwerkbandbreite, Datenbankverbindungen, Thread-Pool-Auslastung und Warteschlangentiefen. Ohne eine akkurate Baseline sind alle zukünftigen Prognosen Ratewerk.
- Korrelieren Sie Ressourcenverbrauch mit Geschäftskennzahlen wie aktiven Nutzern, Transaktionen pro Stunde oder aufgenommenem Datenvolumen. Dieses Korrelationsmodell erlaubt es Ihnen, Geschäftswachstumsprognosen in konkrete Ressourcenanforderungen zu übersetzen, statt sich auf willkürliche Multiplikatoren zu verlassen.
- Identifizieren Sie die aktuellen Sättigungspunkte des Systems durch Lasttests oder Analyse historischer Vorfälle. Bestimmen Sie, welche Ressource unter zunehmender Last zuerst erschöpft — dies ist die bindende Einschränkung, die Ausfälle auslösen wird. In Legacy-Systemen sind diese Einschränkungen oft Datenbankverbindungen, Thread-Pools oder Speicher, nicht CPU.
- Bauen Sie ein einfaches Kapazitätsmodell, das prognostiziertes Arbeitslastwachstum auf Ressourcenbedarf abbildet. Selbst ein tabellenkalkulationsbasiertes Modell, das aktuelle Trends extrapoliert und bekannte Kosten pro Transaktion anwendet, bietet weit mehr Einsicht als gar kein Modell. Aktualisieren Sie das Modell vierteljährlich mit frischen Messungen.
- Definieren Sie Kapazitätsschwellen und Alarmgrenzen bei 70 % und 85 % Auslastung für jede kritische Ressource. Diese Schwellen geben Betriebsteams Vorlaufzeit zum Handeln, bevor Sättigung nutzersichtbare Probleme verursacht. In Legacy-Systemen, wo Skalierung langsam oder manuell ist, sind frühere Warnschwellen essenziell.
- Planen Sie explizit für Spitzenperioden, indem Sie historische Muster wie Monatsendverarbeitung, saisonale Traffic-Spitzen oder Batch-Job-Terminkonflikte analysieren. Legacy-Systeme haben häufig Batch-Arbeitslasten, die mit interaktivem Traffic konkurrieren, und Kapazitätspläne müssen berücksichtigen, dass beide gleichzeitig laufen.
- Dimensionieren Sie Worker-Pools, Thread-Pools und Verbindungspools basierend auf gemessenen Ressourcenkosten pro Anfrage und prognostizierter Nebenläufigkeit, nicht auf Standardwerten oder von der initialen Bereitstellung geerbten Werten. Dokumentieren Sie die Begründung hinter jeder Pool-Größe, sodass zukünftige Betreuer Einstellungen anpassen können, während sich Arbeitslasten ändern.
- Integrieren Sie Kapazitätsplanung in den Change-Management-Prozess. Bevor Sie neue Features oder Integrationen zu einem Legacy-System deployen, schätzen Sie deren Ressourcenauswirkung und verifizieren Sie, dass die aktuelle Kapazität sie absorbieren kann. Viele Legacy-Systemausfälle entstehen aus neuen Arbeitslasten, die hinzugefügt wurden, ohne ihre Auswirkung auf bereits eingeschränkte Ressourcen zu berücksichtigen.

## Tradeoffs ⇄

> Kapazitätsplanung verringert das Risiko von Ausfällen und Performance-Verschlechterung, indem Ressourcenbedarf antizipiert wird, erfordert aber laufende Investition in Messung, Modellierung und organisatorische Disziplin.

**Vorteile:**

- Verhindert Ressourcenerschöpfungsausfälle, indem frühe Sichtbarkeit auf sich nähernde Grenzen für Datenbankverbindungen, Thread-Pools, Worker-Prozesse und Warteschlangenkapazität geboten wird.
- Verringert Notfall-Skalierungsvorfälle, indem Teams Wochen oder Monate Vorlaufzeit statt Stunden gegeben werden, was besonders wertvoll für Legacy-Systeme ist, wo Skalierung Beschaffung, Konfiguration oder architektonische Änderungen beinhaltet, die nicht schnell erfolgen können.
- Ermöglicht informierte Entscheidungen über Hardware-Investitionen, Cloud-Ressourcenbereitstellung und architektonische Refaktorierung, indem sie auf gemessenen Daten statt Intuition gegründet werden.
- Verbessert die Zuverlässigkeit während Spitzenperioden, indem sichergestellt wird, dass das System für bekannte Nachfragespitzen bereitgestellt ist, statt von vorhersehbaren saisonalen oder Batch-Arbeitslasterhöhungen überrascht zu werden.
- Schafft institutionelles Wissen über Systemgrenzen und Wachstumsmuster, das bestehen bleibt, selbst wenn sich Teammitglieder ändern, was für Legacy-Systeme mit begrenzter Dokumentation kritisch ist.

**Kosten und Risiken:**

- Erfordert Instrumentierung und Monitoring, die im Legacy-System möglicherweise nicht existieren, und das Hinzufügen kann in Codebasen, die nicht für Observability designt wurden, kostspielig sein.
- Kapazitätsmodelle, die auf der aktuellen Architektur basieren, könnten ungültig werden, wenn das System refaktoriert, migriert oder erheblich geändert wird, was den Neuaufbau des Modells erfordert.
- Übermäßiges Vertrauen in Prognosen kann zu Überprovisionierung führen und Budget für nie genutzte Ressourcen verschwenden, besonders wenn Wachstumsschätzungen optimistisch sind.
- Die Pflege und Aktualisierung des Kapazitätsmodells erfordert laufenden Aufwand von Ingenieuren, die sowohl die Systeminterna als auch den Geschäftskontext verstehen, was mit Feature-Entwicklungszeit konkurriert.
- In Legacy-Systemen mit undurchsichtigen oder schlecht verstandenen Ressourcenverbrauchsmustern könnte der Aufbau eines akkuraten Modells erheblichen Vorabuntersuchungs- und Profiling-Aufwand erfordern.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Kapazitätsplanung Ressourcenmanagementprobleme in Legacy-Systemen adressiert.

Ein Finanzdienstleistungsunternehmen betreibt ein 15 Jahre altes Transaktionsverarbeitungssystem, das Zahlungsabwicklungen handhabt. Das System nutzt einen festen Pool von 50 Datenbankverbindungen und 32 Worker-Threads, Werte, die beim initialen Deployment gesetzt und nie überprüft wurden. Das Transaktionsvolumen ist Jahr für Jahr um 8 % gewachsen, und das Team beginnt intermittierende Verbindungserschöpfung während der Monatsend-Abwicklungsläufe zu erleben. Durch die Etablierung eines Kapazitätsmodells, das Transaktionsvolumen mit Verbindungsnutzung und Thread-Auslastung korreliert, bestimmt das Team, dass sie ihre Verbindungspool-Kapazität bei aktuellen Wachstumsraten innerhalb von sechs Monaten dauerhaft überschreiten werden. Sie erhöhen proaktiv die Pool-Größen, fügen Connection Pooling via pgBouncer hinzu und planen Batch-Abwicklungen so, dass sie Überlappung mit interaktivem Spitzentraffic vermeiden. Die monatlichen Ausfälle hören auf, und das Kapazitätsmodell wird zu einem ständigen Tagesordnungspunkt in vierteljährlichen Betriebsüberprüfungen.

Eine Logistikplattform verarbeitet Sendungsverfolgungsereignisse durch eine Nachrichtenwarteschlange mit einem festen Satz von Worker-Prozessen. Über zwei Jahre verdoppelt sich die Anzahl der verfolgten Sendungen, aber die Worker-Anzahl bleibt unverändert. Aufgabenwarteschlangen beginnen sich während Versandspitzenstunden aufzustauen, was Verfolgungsupdates um mehrere Stunden verzögert. Nachdem das System zur Messung der Verarbeitungskosten pro Ereignis instrumentiert und die Warteschlangentiefe mit dem Sendungsvolumen korreliert wurde, baut das Team eine Prognose, die zeigt, dass die aktuelle Worker-Kapazität innerhalb von drei Monaten vollständig gesättigt sein wird. Sie implementieren Auto-Scaling-Regeln für Worker-Prozesse, gebunden an Warteschlangentiefen-Schwellen, und etablieren monatliche Kapazitätsüberprüfungen, die tatsächliches Wachstum gegen Prognosen vergleichen. Warteschlangenaufbau-Vorfälle sinken von wöchentlichen Vorkommnissen auf nahezu null, und das Team gewinnt Vertrauen bei der Planung von Infrastrukturbudgets basierend auf gemessenen Daten statt reaktivem Feuerlöschen.
