---
title: Continuous Delivery
description: Automatische Vorbereitung von Software-Änderungen für das Produktiv-Deployment.
category:
- Operations
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/continuous-delivery/
problems:
- manual-deployment-processes
- complex-deployment-process
- long-build-and-test-times
- long-release-cycles
- deployment-risk
- large-risky-releases
- release-anxiety
- deployment-coupling
- deployment-environment-inconsistencies
- frequent-hotfixes-and-rollbacks
- release-instability
- missing-rollback-strategy
- extended-cycle-times
- increased-time-to-market
- immature-delivery-strategy
- flaky-tests
- increased-manual-testing-effort
- mixed-coding-styles
- outdated-tests
- tool-limitations
- long-lived-feature-branches
- style-arguments-in-code-reviews
- customization-outside-version-control
layout: solution
lang: de
en_slug: ci-cd-pipeline
related_solutions:
- slug: continuous-deployment
  similarity: 0.9
- slug: blue-green-canary-deployments
  similarity: 0.85
- slug: continuous-delivery
  similarity: 0.85
- slug: continuous-integration-and-delivery
  similarity: 0.8
- slug: documentation-as-code
  similarity: 0.8
- slug: test-coverage-strategy
  similarity: 0.8
---

## Description

Eine CI/CD-Pipeline automatisiert Build, Test und Deployment einer Änderung vom Commit bis zur Produktion, was einen manuellen, ritualbasierten Release-Prozess durch einen ersetzt, der wiederholbar, auditierbar ist und nicht von wenigen Personen abhängt, die zufällig die Schritte auswendig kennen. Legacy-Systeme sind die Umgebungen, in denen dieses manuelle Ritual am tiefsten verwurzelt und am fragilsten ist: Deployment-Wissen konzentriert in ein oder zwei Personen kurz vor dem Ruhestand, Umgebungsunterschiede, die niemand dokumentiert hat, und ein Release-Prozess so riskant, dass Teams ihn monatelang verschieben, was den eventuellen Release nur noch riskanter macht. Die Pipeline zu bauen, indem zunächst der bestehende manuelle Prozess vollständig dokumentiert und dann Schritt für Schritt automatisiert wird — Build und Test zuerst, dann Datenbankmigrationen, dann automatisierter Rollback —, verwandelt Deployment von einem gefürchteten Ereignis in eine routinemäßige, risikoarme Operation, obwohl dies für eine genuin gealterte Legacy-Umgebung echte Investition erfordert, um Umgebungsinkonsistenzen zu schließen, die sich über Jahre angehäuft haben.

## How to Apply ◆

> Legacy-Systeme verlassen sich typischerweise auf manuelle, ritualbasierte Deployment-Prozesse, die Wissen bei wenigen Personen konzentrieren und jeden Release zu einem hochriskanten Ereignis machen; die Einführung einer CI/CD-Pipeline ersetzt dieses fragile Ritual durch einen wiederholbaren, auditierbaren automatisierten Prozess.

- Beginnen Sie damit, den aktuellen manuellen Deployment-Prozess in erschöpfendem Detail zu dokumentieren, bevor Sie irgendetwas automatisieren. Jeder Schritt, jedes von Hand ausgeführte Skript, jede auf dem Server bearbeitete Konfigurationsdatei muss erfasst werden. Diese Dokumentation legt verstecktes Deployment-Wissen offen und dient als Spezifikation dafür, was die Pipeline reproduzieren muss.
- Automatisieren Sie zuerst die Build- und Unit-Test-Stufe und machen Sie sie zum verpflichtenden Gate des Teams vor jedem Code-Review. Selbst in einem Legacy-System mit spärlicher Testabdeckung hat das automatische Abfangen von Kompilierungsfehlern und grundlegenden Fehlschlägen innerhalb von Minuten nach einem Commit unmittelbaren Wert.
- Behandeln Sie die Deployment-Skripte und Konfiguration des Legacy-Systems als Code: Verschieben Sie sie in die Versionskontrolle, wenden Sie Code-Review an, und unterwerfen Sie die Pipeline-Definition selbst denselben Standards wie Produktionscode. Dies eliminiert das Single-Point-of-Failure-Wissensproblem, das Legacy-Deployments plagt.
- Investieren Sie stark in Umgebungsparität. Legacy-Systeme, die über Jahre manuell verwaltet wurden, haben oft undokumentierte Unterschiede zwischen Produktion und jeder niedrigeren Umgebung. Schließen Sie diese Lücken systematisch, weil Deployments, die in Staging funktionieren, aber in Produktion scheitern, das Vertrauen in die gesamte Pipeline untergraben.
- Automatisieren Sie Datenbankschemamigrationen unter Nutzung eines Versionierungswerkzeugs und integrieren Sie sie in die Pipeline. Legacy-Datenbankschemata häufen häufig manuelle Änderungen an, die direkt in Produktion angewendet wurden; ein Migrationswerkzeug macht diese Änderungen reproduzierbar, umkehrbar und auditierbar.
- Implementieren Sie automatisierten Rollback als erstklassige Pipeline-Operation von Anfang an. Legacy-Systeme haben oft die längsten Rollback-Prozeduren und die höchste Rollback-Angst; Rollback zu automatisieren und ihn regelmäßig zu üben, bevor eine Krise es verlangt, ist essenziell.
- Nutzen Sie die Pipeline, um schrittweise Testabdeckung aufzubauen: Jeder neue Integrationstest oder End-to-End-Test, der hinzugefügt wird, um ein Legacy-Verhalten zu validieren, erhöht das Vertrauen, das die Pipeline bietet, und verringert die manuelle Verifikationslast bei jedem Release.
- Wenden Sie Canary- oder Blue-Green-Deployment-Strategien für hochriskante Legacy-Änderungen an, indem ein Bruchteil des Produktions-Traffics zur neuen Version geleitet wird, bevor der vollständige Umstieg erfolgt. Dies gibt Teams einen sicheren Mechanismus für Releases, die zuvor Ausfallzeitfenster außerhalb der Geschäftszeiten erforderten.

## Tradeoffs ⇄

> Eine CI/CD-Pipeline verwandelt Legacy-Deployments von unvorhersehbaren manuellen Ereignissen in kontrollierte, beobachtbare Prozesse, aber die Vorabinvestition ist erheblich und der kulturelle Wandel ist bedeutsam.

**Vorteile:**

- Verringert den Explosionsradius jedes Releases, indem kleine, häufige Deployments statt großer, seltener Batch-Releases ermöglicht werden, bei denen Fehlerdiagnose schwierig ist.
- Eliminiert Deployment-Wissenskonzentration: Wenn die Pipeline der einzige Pfad zur Produktion ist, kann jedes Teammitglied ein Deployment sicher auslösen, was die Abhängigkeit von den wenigen Personen entfernt, die das manuelle Ritual kennen.
- Bietet eine auditierbare Deployment-Historie, die jede Produktionsänderung mit einem spezifischen Commit, Build und Testergebnis verbindet — wertvoll für Compliance-Anforderungen, die in Legacy-Systemumgebungen üblich sind.
- Verkürzt die Feedback-Schleife zwischen einer Codeänderung und ihrer Validierung in einer produktionsähnlichen Umgebung, was den wochenlangen Zyklus ersetzt, der für manuelle Legacy-Release-Prozesse typisch ist.
- Macht Rollback zu einer routinemäßigen, geübten Operation statt einer Notfallprozedur, was die mittlere Wiederherstellungszeit dramatisch verringert, wenn ein Deployment eine Regression einführt.

**Kosten und Risiken:**

- Legacy-Systeme haben oft tiefe Umgebungsinkonsistenzen, benutzerdefinierte Serverkonfigurationen und undokumentierte Abhängigkeiten, die die Pipeline-Einrichtung erheblich komplexer machen als für Greenfield-Systeme.
- Das Erreichen von Umgebungsparität für Legacy-Systeme könnte erhebliche Infrastrukturinvestition erfordern, besonders wenn Produktion auf Hardware- oder Betriebssystemversionen läuft, die in niedrigeren Umgebungen schwer zu replizieren sind.
- Die bestehende Test-Suite eines Legacy-Systems ist häufig zu spärlich und zu langsam, um als verlässliches Pipeline-Gate zu dienen, ohne erhebliche Investition in Testabdeckung und Performance.
- Teams, die an manuelle Deployments gewöhnt sind, könnten sich gegen die Disziplin wehren, alle Änderungen durch die Pipeline zu leiten, besonders unter Druck, was das Risiko von Out-of-Band-Produktionsänderungen schafft, die den Prüfpfad der Pipeline untergraben.
- Pipeline-Pflege wird zu einer neuen laufenden Verantwortung: Pipeline-Konfigurationen, Container-Images und Umgebungsdefinitionen altern zusammen mit dem Legacy-Code und erfordern regelmäßige Updates, um funktionsfähig zu bleiben.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie CI/CD-Pipelines die Deployment-Fragilität angehen, die sich in langlebigen Legacy-Systemen anhäuft.

Eine mittelgroße Bank, die ein Ende-der-1990er-Java-Stack-basiertes Kernkontoverwaltungssystem betrieb, releaste vierteljährlich Software unter Nutzung eines sechsköpfigen Release-Teams, das ein 200-Schritte-manuelles Runbook über ein Wochenende ausführte. Releases liefen regelmäßig über die Zeit hinaus, was Montagmorgen-Notfallfixes erforderte, und das für die Ausführung des Runbooks benötigte Wissen lag fast vollständig bei zwei Senior-Ingenieuren, die sich dem Ruhestand näherten. Die Organisation investierte sechs Monate in den Bau einer Jenkins-Pipeline, die Build, Datenbankmigration, Deployment und Smoke-Tests für jeden Release automatisierte. Der erste automatisierte Release lief in vierzig Minuten verglichen mit den vorherigen sechzehn Stunden, und innerhalb eines Jahres war der Release-Rhythmus auf monatlich gestiegen, mit Plänen für zweiwöchentliche Lieferung. Die beiden Senior-Ingenieure übertrugen ihr Deployment-Wissen in Pipeline-Konfiguration und Dokumentation statt es als institutionelles Gedächtnis zu tragen.

Eine Regierungsbehörde, die ein Legacy-Genehmigungsverarbeitungssystem verwaltete, hatte seit achtzehn Monaten nicht in Produktion releast, weil ein vorheriges Deployment eine kritische Datenbanktabelle korrumpiert hatte und zwei Wochen zur Wiederherstellung gebraucht hatte. Das Trauma hatte das Team tief zögerlich gemacht, Produktion erneut anzufassen. Die Behörde engagierte einen DevOps-Berater, der damit begann, die exakten Schritte zu erfassen, die zur Korruption geführt hatten — eine manuelle Migration in falscher Reihenfolge — und eine Flyway-basierte Migrationspipeline zu bauen, die die Migrationsreihenfolge automatisch durchsetzte. Das Team baute dann eine Staging-Umgebung mit einer aktuellen anonymisierten Kopie der Produktionsdaten und führte die vollständige Deployment-Pipeline wöchentlich dagegen aus, selbst wenn kein Release geplant war, und demonstrierte wiederholt, dass die Pipeline sicher war. Nach drei Monaten erfolgreicher Staging-Deployments releaste das Team in Produktion. Das zuvor gefürchtete Ereignis war unspektakulär.

Ein Einzelhandelsunternehmen, das ein Legacy-Auftragsmanagementsystem betrieb, deployte neue Versionen, indem es sich per SSH in Produktionsserver einloggte und manuell JAR-Dateien während verkehrsarmer Perioden um 2 Uhr morgens ersetzte. Der Prozess erforderte mindestens zwei Ingenieure, die beide die exakte Schrittfolge für jedes Deployment kannten. Als einer dieser Ingenieure das Unternehmen verließ, wurde der verbleibende Ingenieur zu einem Deployment-Single-Point-of-Failure. Um dies anzugehen, containerisierte das Unternehmen die Legacy-Anwendung — ohne ihre Logik zu ändern — und baute eine GitLab-CI-Pipeline, die bei jedem Merge zum Main-Branch ein unveränderliches Docker-Image baute, die bestehenden Integrationstests gegen dieses Image ausführte und es in eine Registry pushte. Deployments verschoben sich von SSH-Sitzungen zu einem einzigen Pipeline-Auslöser, das Deployment-Wissen wurde explizit in der Pipeline-Konfiguration, und zum ersten Mal konnte jeder Entwickler im Team das Legacy-System sicher deployen.
