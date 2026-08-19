---
title: Observability
description: Umsetzung strukturierten Loggings, verteilten Tracings und Metriken.
category:
- Operations
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/observability/
problems:
- monitoring-gaps
- debugging-difficulties
- slow-incident-resolution
- constant-firefighting
- system-outages
- gradual-performance-degradation
- log-spam
- excessive-logging
- logging-configuration-issues
- insufficient-audit-logging
- log-injection-vulnerabilities
- slow-application-performance
- single-points-of-failure
- cascade-failures
- unpredictable-system-behavior
- increased-error-rates
- database-connection-leaks
- cache-invalidation-problems
- incorrect-max-connection-pool-size
- index-fragmentation
- misconfigured-connection-pools
- resource-allocation-failures
- service-discovery-failures
- task-queues-backing-up
- unreleased-resources
layout: solution
lang: de
en_slug: observability-and-monitoring
related_solutions:
- slug: logging
  similarity: 0.75
- slug: distributed-tracing
  similarity: 0.75
- slug: service-level-objectives
  similarity: 0.75
- slug: status-monitoring
  similarity: 0.75
- slug: error-logs
  similarity: 0.7
- slug: logging-guidelines
  similarity: 0.7
---

## Description

Observability instrumentiert ein System mit strukturiertem Logging, Korrelations-IDs, verteiltem Tracing und Metriken, damit neue Fragen über sein Verhalten im Nachhinein gestellt und beantwortet werden können, statt nur Fehlermodi zu erkennen, die das Team bereits zuvor gesehen hat. Legacy-Systeme profitieren davon überproportional, weil sie typischerweise mit der geringsten Sichtbarkeit jedes Systems in der Organisation betrieben wurden — Freitext-Logs, verstreut über Server, nur per SSH erreichbar, keine Korrelation zwischen einem nutzersichtbaren Symptom und seiner Grundursache irgendwo im Stack —, was genau der Grund ist, warum Vorfalluntersuchung in diesen Systemen so oft Stunden manueller Log-Korrelation braucht statt einer einzigen Abfrage. Selbst grundlegendes strukturiertes Logging mit einer Korrelations-ID an der Systemgrenze nachzurüsten ist meist die eine Änderung mit dem höchsten Wert, obwohl das Nachrüsten selbst über eine große, inkonsistent instrumentierte Codebasis hinweg echt arbeitsintensiv ist, und unverwaltetes Telemetrievolumen kann selbst zu erheblichen neuen Betriebskosten werden.

## How to Apply ◆

> Observability zu einem Legacy-System hinzuzufügen bedeutet, die Fähigkeit aufzubauen, neue Fragen über Systemverhalten zu stellen — beginnend mit den Fehlermodi, die das Team bereits erlebt hat, aber derzeit nicht erklären kann.

- Beginnen Sie mit strukturiertem Logging. Ersetzen oder ergänzen Sie Legacy-Freitext-Log-Ausgaben mit maschinell parsbaren strukturierten Log-Einträgen (JSON), die mindestens enthalten: Zeitstempel, Schweregrad, Dienst-/Komponentenname, Korrelations-ID und eine kurze Ereignisbeschreibung. Viele Legacy-Systeme protokollieren Text in Dateien ohne strukturierte Felder — selbst das Hinzufügen eines Korrelations-ID-Feldes transformiert die diagnostische Fähigkeit.
- Führen Sie Korrelations-IDs an der Systemgrenze ein — dem Einstiegspunkt, an dem Anfragen ankommen. Geben Sie diese ID durch alle nachgelagerten Aufrufe weiter, einschließlich Aufrufe an andere Legacy-Dienste, Datenbanken und Message Queues. In Legacy-Systemen, die mehrere Codebasen umspannen, erfordert dies Koordination, ist aber die wertvollste mögliche Observability-Verbesserung.
- Fügen Sie die vier goldenen Signale als Metriken für jede Hauptkomponente hinzu: Latenz (mit Perzentilaufschlüsselungen, nicht nur Durchschnitten), Anfrage-/Transaktionsrate, Fehlerrate und Ressourcensättigung. Passen Sie diese für Legacy-Batch-Systeme an batch-geeignete Signale an: verarbeitete Datensätze pro Lauf, Fehlerraten und Verarbeitungsverzug.
- Nutzen Sie OpenTelemetry, wo möglich, um Anbieter-Lock-in zu vermeiden. Viele Legacy-Frameworks (Java Spring, .NET, ältere Python- und Ruby-Frameworks) haben OpenTelemetry-Agenten oder SDKs, die automatische Instrumentierung mit minimalen Codeänderungen bieten — HTTP-Aufrufe, Datenbankabfragen und gängige Messaging-Bibliotheken sind oft abgedeckt, ohne Instrumentierungscode zu schreiben.
- Priorisieren Sie zuerst die Instrumentierung von Integrationspunkten. Legacy-Systeme versagen typischerweise an Grenzen — beim Aufruf von Drittanbieter-APIs, beim Lesen aus gemeinsamen Datenbanken, beim Konsumieren aus Queues. Dies sind die schwierigsten Stellen zum Debuggen ohne Traces und die Stellen, an denen Sichtbarkeit den unmittelbarsten Wert liefert.
- Etablieren Sie früh Service Level Objectives (SLOs), selbst wenn informell. Ohne eine Definition akzeptablen Verhaltens sind Monitoring-Schwellenwerte willkürlich, und Alarmmüdigkeit ist unvermeidlich. SLOs fokussieren die Aufmerksamkeit des Teams auf die Signale, die tatsächlich Nutzerauswirkung repräsentieren.
- Richten Sie eine zentralisierte Log-Aggregations- und Abfrageplattform ein (Elasticsearch/Kibana, Loki/Grafana oder ein kommerzielles Äquivalent). Legacy-Systeme haben oft Logs, verteilt über Dutzende Server, nur per SSH zugänglich. Ihre Zentralisierung verwandelt Vorfalluntersuchung von einer mehrstündigen Mehrserver-Übung in eine einzige Abfrage.
- Instrumentieren Sie geschäftliche Metriken neben technischen Metriken. Legacy-Systeme haben oft nicht offensichtliche geschäftliche Invarianten (Auftragszahlen pro Minute, Transaktionsgenehmigungsraten, Batch-Fertigstellungszeiten), die aussagekräftigere Indikatoren für Systemgesundheit sind als CPU-Nutzung.

## Tradeoffs ⇄

> Observability wandelt den Betrieb von Legacy-Systemen von reaktivem Feuerwehreinsatz zu datengestützter Diagnose, aber das Nachrüsten von Instrumentierung in bestehende Systeme ist eine anhaltende Engineering-Investition statt einer einmaligen Änderung.

**Vorteile:**

- Die Vorfalluntersuchungszeit sinkt erheblich, sobald Korrelations-IDs und strukturierte Logs vorhanden sind. Was zuvor SSH-Zugriff auf mehrere Server und manuelles Durchsuchen von Log-Dateien erforderte, wird zu einer einzigen Abfrage über ein zentralisiertes Log-System.
- Teams gewinnen die Fähigkeit, neuartige Fehlermodi ohne Vorwissen über diesen spezifischen Fehler zu diagnostizieren. Legacy-Systeme produzieren regelmäßig überraschendes Verhalten; Observability ermöglicht die Untersuchung von Überraschungen statt nur die Erkennung zuvor gesehener Muster.
- Observability-Daten verringern die Abhängigkeit vom Erfahrungswissen leitender Ingenieure, die das Verhalten des Systems mental modelliert haben. Junior-Teammitglieder können Vorfälle unabhängig untersuchen, mit denselben Daten, die die Leitenden nutzen.
- Performance-Engpässe in Legacy-Systemen — oft unbekannt, weil keine Messung existierte — werden durch verteilte Traces sichtbar. Teams entdecken häufig, dass Komponenten, von denen sie annahmen, sie seien schnell, für erhebliche Latenz verantwortlich sind.
- Eine während der Modernisierung gebaute Observability-Schicht liefert das Verifikationssignal für jede Änderung: Teams können empirisch bestätigen, dass ein Refactoring das Systemverhalten nicht verändert hat, statt sich auf Testabdeckung zu verlassen, die Legacy-Systemen oft fehlt.

**Kosten und Risiken:**

- Instrumentierung in eine etablierte Legacy-Codebasis nachzurüsten ist arbeitsintensiv, besonders in Systemen ohne bestehende Logging-Konventionen. Das Hinzufügen von Korrelations-ID-Weitergabe über eine große, schlecht strukturierte Codebasis kann Hunderte kleiner Änderungen erfordern.
- Telemetrie-Datenvolumen für Legacy-Systeme unter erheblicher Last kann enorm sein. Ungesampelte verteilte Traces, hochkardinale Metriken und ausführliche strukturierte Logs erzeugen Speicher- und Verarbeitungskosten, die geplant werden müssen. Unverwaltet können diese Kosten so groß sein wie die Anwendungsinfrastruktur selbst.
- Legacy-Systeme nutzen oft ältere Frameworks und Bibliotheken mit begrenzter oder keiner OpenTelemetry-Unterstützung. Benutzerdefinierte Instrumentierung muss geschrieben werden, was Entwicklungszeit hinzufügt und Code erzeugt, der neben der Legacy-Codebasis gepflegt werden muss.
- Alarmmüdigkeit ist ein besonderes Risiko beim Hinzufügen von Monitoring zu einem zuvor unüberwachten System. Anfängliche Alarmschwellenwerte sind oft falsch und produzieren Fluten falsch-positiver Ergebnisse, die Teams zu ignorieren lernen — einschließlich Alarme, die schließlich echte Probleme darstellen.
- Teams brauchen Schulung in Observability-Werkzeugen und Untersuchungstechniken. Das Hinzufügen von Grafana-Dashboards und einer Jaeger-Instanz verbessert die Vorfallreaktion nicht automatisch, wenn das Team nicht weiß, wie man sie unter Druck effektiv nutzt.

## How It Could Be

> Legacy-Systeme sind oft die Umgebungen, in denen Observability den unmittelbarsten betrieblichen Wert liefert, gerade weil sie historisch mit der geringsten Sichtbarkeit betrieben wurden.

Das Produktionsplanungssystem eines Fertigungsunternehmens — ein zwölf Jahre alter Java-Monolith — erlebte intermittierende Verlangsamungen, die die Werksplanung für unvorhersehbare Zeiträume von zehn bis vierzig Minuten verschlechterten. Das Betriebsteam hatte keine Instrumentierung über grundlegende CPU- und Speicherdiagramme hinaus, und Untersuchungen endeten immer mit „das System hat sich von selbst erholt". Nach dem Hinzufügen strukturierten Loggings mit Korrelations-IDs und der Integration von OpenTelemetrys Java-Agent für automatische Instrumentierung entdeckte das Team innerhalb weniger Tage, dass die Verlangsamungen mit einer spezifischen Kombination von Datenbankabfragemustern korrelierten, die auftrat, wenn zwei bestimmte geplante Jobs gleichzeitig liefen. Die Jobs hatten immer gleichzeitig gelaufen, aber die Verlangsamung manifestierte sich erst, als das Produktionsdatenvolumen einen Schwellenwert überschritt, der im Vorjahr überschritten worden war. Ohne die verteilten Traces wäre diese Verbindung jahrelang unsichtbar geblieben.

Ein großes Versicherungsunternehmen betrieb ein COBOL-basiertes Schadenverarbeitungssystem neben einer modernen Java-Middleware-Schicht, die zwischen dem Mainframe und webseitigen Diensten übersetzte. Vorfälle an der Integrationsgrenze waren häufig, und jeder erforderte, dass ein Spezialistenteam Zeitstempel manuell über Mainframe-SYSOUT-Logs und Java-Anwendungslogs hinweg korrelierte — ein Prozess, der Stunden dauerte. Das Team führte strukturiertes Logging und Korrelations-IDs in der Java-Middleware-Schicht ein und baute eine Log-Weiterleitungspipeline, die die Job-Abschluss-Aufzeichnungen des COBOL-Systems einbezog. Plötzlich konnte eine einzige Abfrage über die aggregierten Logs die vollständige Geschichte eines Schadenverarbeitungsausfalls von der Web-Anfrage über die Java-Middleware bis zur Mainframe-Job-Ausführung zeigen. Die durchschnittliche Zeit bis zur Diagnose von Integrationsvorfällen fiel von vier Stunden auf unter dreißig Minuten.

Eine Finanzhandelsplattform, die eine Mischung aus Legacy-C++-Komponenten und neueren Python-Diensten betrieb, hatte mit unvorhersehbaren Latenzspitzen bei Markteröffnung zu kämpfen. Teams beschuldigten in jedem Vorfall unterschiedliche Komponenten. Nach der Instrumentierung des Systems mit Prometheus-Metriken und Grafana-Dashboards, fokussiert auf die vier goldenen Signale für jede Komponente, zeigte sich ein Muster, das zuvor nicht sichtbar gewesen war: ein Legacy-C++-Auftrags-Router sättigte seinen Verbindungspool während der ersten zwei Minuten der Markteröffnung, was Gegendruck verursachte, der sich durch die neueren Dienste auf Weisen ausbreitete, die wie unabhängige Ausfälle in den eigenen Metriken jedes Dienstes aussahen. Die Behebung erforderte Änderungen an der Verbindungspool-Konfiguration der Legacy-C++-Komponente — eine einzeilige Änderung, die niemand in drei Jahren Untersuchungen gefunden hatte, weil niemand die Daten hatte, die dorthin wiesen.
