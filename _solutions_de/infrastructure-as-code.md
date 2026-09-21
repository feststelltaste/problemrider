---
title: Infrastructure as Code
description: Definition und Verwaltung von Infrastruktur durch Code.
category:
- Operations
quality_tactics_url: https://qualitytactics.de/en/maintainability/infrastructure-as-code/
problems:
- configuration-drift
- configuration-chaos
- deployment-environment-inconsistencies
- environment-variable-issues
- inadequate-configuration-management
- legacy-configuration-management-chaos
- manual-deployment-processes
- complex-deployment-process
- deployment-risk
- poor-system-environment
- poor-operational-concept
- operational-overhead
- tool-limitations
- testing-environment-fragility
- customization-outside-version-control
layout: solution
lang: de
en_slug: infrastructure-as-code
related_solutions:
- slug: immutable-infrastructure
  similarity: 0.8
- slug: ci-cd-pipeline
  similarity: 0.75
- slug: documentation-as-code
  similarity: 0.75
- slug: secret-management
  similarity: 0.7
- slug: code-review-process-reform
  similarity: 0.7
- slug: static-analysis-and-linting
  similarity: 0.7
---

## Description

Infrastructure as Code definiert Server, Netzwerke und Konfiguration als versionierte Definitionen, die die Umgebung reproduzierbar bereitstellen, und ersetzt Jahre manueller, undokumentierter Änderungen, die direkt von wem auch immer den Betrieb an diesem Tag gerade handhabte, vorgenommen wurden. Legacy-Infrastruktur ist der Ort, an dem diese manuelle Anhäufung am schwersten und gefährlichsten ist: Umgebungen, die auf Weisen auseinandergedriftet sind, die niemand vollständig versteht, Ressourcen, deren ursprünglichen Zweck nur ein inzwischen ausgeschiedener Ingenieur je kannte, und ein „funktioniert in Produktion, aber nicht in Staging"-Problem, das vollständig in dieser Drift wurzelt. Bestehende Ressourcen schrittweise in IaC zu importieren, beginnend mit dem, was sich am häufigsten ändert und die meisten Vorfälle verursacht, wandelt diesen unsichtbaren und undokumentierten Zustand in etwas Auditierbares und Reproduzierbares um — obwohl die Migration selbst echtes Risiko birgt, da eine beschädigte State-Datei oder ein unachtsamer Import so disruptiv sein kann wie die Probleme, die sie beheben soll.

## How to Apply ◆

> Infrastructure as Code auf ein Legacy-System anzuwenden bedeutet, Jahre angesammelter manueller Konfiguration und Erfahrungswissen in versionierte, auditierbare Definitionen umzuwandeln — beginnend mit den flüchtigsten und am wenigsten verstandenen Teilen der Infrastruktur.

- Beginnen Sie mit einem Infrastruktur-Audit: Kartieren Sie, was tatsächlich über Produktions-, Staging- und Entwicklungsumgebungen existiert. In Legacy-Systemen sind diese Umgebungen typischerweise weit auseinandergedriftet. Dokumentieren Sie die Unterschiede, bevor Sie irgendeinen Code schreiben — die Lücken offenbaren, wo die höchsten Risiken liegen.
- Übernehmen Sie eine inkrementelle Importstrategie statt einer Big-Bang-Neuerstellung. Nutzen Sie Werkzeuge wie den `import`-Befehl von Terraform, um bestehende manuell bereitgestellte Ressourcen eine Ressourcengruppe nach der anderen unter IaC-Verwaltung zu bringen, beginnend mit der Infrastruktur, die sich am häufigsten ändert und die meisten Vorfälle verursacht.
- Priorisieren Sie Umgebungsparität. Legacy-Systeme haben typischerweise ein „funktioniert in Produktion, aber nicht in Staging"-Problem, das in Konfigurationsdrift wurzelt. Sobald die kritischste Infrastruktur kodifiziert ist, nutzen Sie dieselben IaC-Definitionen für alle Umgebungen und beseitigen manuelle umgebungsspezifische Überschreibungen.
- Speichern Sie alle IaC-Definitionen in Versionskontrolle neben dem Anwendungscode. Für Legacy-Systeme ohne vorherige Versionskontrolldisziplin für Infrastruktur ist selbst ein einfaches Git-Repository für Terraform-Dateien eine transformative Verbesserung gegenüber gemeinsamen Tabellenkalkulationen und Runbook-Wikis.
- Erzwingen Sie Peer-Review für alle Infrastrukturänderungen mittels Pull Requests, einschließlich eines Plan-Review-Schritts. Die Ausgabe von `terraform plan` muss überprüft werden, bevor ein `apply` läuft. Diese einzige Änderung fängt die Klasse versehentlicher Löschungen und Fehlkonfigurationen ab, unter denen Legacy-Systeme routinemäßig leiden.
- Erfassen Sie das „Warum" in Commit-Nachrichten und Kommentaren. Legacy-Infrastruktur enthält oft Firewall-Regeln, Instanzgrößenwahlen und Netzwerkkonfigurationen, deren ursprüngliche Begründung unbekannt ist. Fügen Sie beim Kodifizieren bestehender Infrastruktur erklärende Kommentare hinzu, die erfassen, was über die Überlegung bekannt ist — selbst wenn das nur „das war schon da, und wir wissen nicht warum" ist.
- Trennen Sie Infrastruktur-State-Dateien nach Explosionsradius. Legen Sie nicht die gesamte Infrastruktur des Legacy-Systems in eine State-Datei. Trennen Sie Netzwerk-, Rechen-, Datenbank- und Anwendungsschichten, sodass ein Fehler in einem Bereich keine Zerstörung in einem anderen auslösen kann.
- Fügen Sie statische Analyse und Sicherheitsscanning (Werkzeuge wie `tflint`, `checkov`) von Anfang an zur CI-Pipeline hinzu. Legacy-Infrastruktur birgt oft Sicherheitsfehlkonfigurationen, die seit Jahren bestehen; automatisiertes Scanning macht diese sichtbar, ohne ein manuelles Sicherheitsaudit zu erfordern.

## Tradeoffs ⇄

> Infrastructure as Code verwandelt Legacy-Umgebungen von unsichtbar und undokumentiert in auditierbar und reproduzierbar, aber der Migrationspfad birgt echtes Risiko, das sorgfältig verwaltet werden muss.

**Vorteile:**

- Konfigurationsdrift — das prägende Betriebsproblem von Legacy-Systemen — wird beseitigt oder sichtbar gemacht. Umgebungen konvergieren, weil sie alle von denselben Quelldefinitionen abgeleitet sind.
- Infrastrukturänderungen gewinnen einen vollständigen Audit-Trail durch die Versionskontrollhistorie: wer was wann und warum geändert hat. Dies ist essenziell für Compliance in regulierten Legacy-Umgebungen, wo Änderungsaufzeichnungen erforderlich sind, aber historisch manuell gepflegt wurden.
- Disaster Recovery wird glaubwürdig. Legacy-Systemen fehlen häufig getestete Wiederherstellungsverfahren; IaC bietet die Fähigkeit, ganze Umgebungen aus Code statt aus institutionellem Gedächtnis und forensischer Untersuchung zu rekonstruieren.
- Infrastrukturwissen entkommt den Köpfen der wenigen Personen, die die aktuelle Umgebung manuell bereitgestellt haben. Die IaC-Definitionen dienen als ausführbare Dokumentation, die jedes Teammitglied lesen und ausführen kann.
- Wiederverwendbare Module ermöglichen Konsistenz über mehrere Legacy-System-Umgebungen hinweg und verringern das Risiko, neue Konfigurationsfehler einzuführen, wenn zusätzliche Umgebungen für Test- oder Migrationszwecke hochgefahren werden.

**Kosten und Risiken:**

- Die anfängliche Migration manuell bereitgestellter Legacy-Infrastruktur in IaC ist aufwändig und risikoreich. Das Importieren bestehender Ressourcen in den State ist mühsam, und das Risiko, während der Migration versehentlich laufende Produktionsinfrastruktur zu zerstören, ist real. Teams brauchen kontrollierte Rollout-Pläne und Rollback-Verfahren für die Migration selbst.
- Legacy-Infrastruktur enthält oft Ressourcen, deren Eigentümerschaft und Zweck unklar sind. Unbekannte Ressourcen zu kodifizieren riskiert, undokumentierte Abhängigkeiten zu brechen; sie nicht zu kodifizieren lässt Lücken in der Abdeckung. Diese Mehrdeutigkeit verlangsamt die Einführung.
- State-Datei-Verwaltung führt eine neue Kategorie betrieblichen Risikos ein. Eine beschädigte oder verlorene Terraform-State-Datei für die Produktionsinfrastruktur eines Legacy-Systems kann die Infrastruktur effektiv unverwaltbar machen, bis der State rekonstruiert ist — potenziell disruptiver als die Probleme, die IaC lösen sollte.
- Legacy-Teams mit begrenzter IaC-Erfahrung stehen vor einer erheblichen Lernkurve. Werkzeuge wie Terraform haben ihre eigene Sprache, ihr eigenes State-Modell und ihre eigenen Fehlermodi. In Teams, die bereits durch Wartungsarbeit ausgelastet sind, braucht Schulung Zeit, die nicht immer verfügbar ist.
- Compliance- und Change-Management-Prozesse in Legacy-Organisationen könnten formale Genehmigungs-Workflows erfordern, die sich nicht sauber auf Pull-Request-basierte IaC-Praktiken abbilden lassen. Die Vereinbarkeit von IaC-Geschwindigkeit mit Anforderungen des Change Advisory Board ist ein häufiger Reibungspunkt.

## How It Could Be

> Die Organisationen, die am meisten von Infrastructure as Code auf Legacy-Systemen profitieren, sind jene, wo Jahre manueller Bereitstellung Umgebungen produziert haben, die niemand vollständig versteht.

Ein Einzelhandelsunternehmen, das eine jahrzehntealte E-Commerce-Plattform betrieb, hatte mehrere hundert EC2-Instanzen, Dutzende Sicherheitsgruppen und Hunderte Datenbankparameterkonfigurationen angesammelt — alle über die Jahre manuell von einer wechselnden Besetzung von Betriebsingenieuren bereitgestellt. Als ein Schlüsselingenieur ging, erkannte das Team, dass sie grundlegende Fragen über ihre eigene Infrastruktur nicht sicher beantworten konnten: welche Instanzen welche Funktionen bedienten, welche Sicherheitsgruppenregeln noch gebraucht wurden und warum bestimmte Instanzen spezifische Instanztypen nutzten. Sie begannen eine systematische IaC-Migration, importierten Ressourcen in den Terraform-State und dokumentierten dabei deren Zweck. Das Audit, das die Migration ihnen aufzwang, offenbarte siebzehn Instanzen, die seit über zwei Jahren ohne klare Eigentümerschaft oder Zweck liefen, und vierzehn übermäßig freizügige Sicherheitsgruppenregeln, die während vergangener Vorfälle hinzugefügt und nie verschärft worden waren.

Eine Organisation des öffentlichen Sektors, die ein Legacy-Fallmanagementsystem betrieb, hatte Umgebungen — Produktion, Vor-Produktion, Test und Entwickler —, die über acht Jahre so stark auseinandergedriftet waren, dass ein im Test verifizierter Bugfix in der Produktion regelmäßig anders funktionierte. Das Team nutzte Terraform, um Produktion als maßgebliche Basislinie zu kodifizieren, und baute dann alle anderen Umgebungen aus denselben Definitionen mit umgebungsspezifischen Parametern neu auf. Innerhalb von drei Monaten war die Vorfallkategorie „funktioniert im Test, schlägt in Produktion fehl" um etwa sechzig Prozent gesunken, und das Team hatte mehrere produktionsspezifische Konfigurationen identifiziert, die Jahre zuvor auf Testumgebungen hätten angewendet werden sollen, es aber nie wurden.

Ein Finanzdienstleistungsunternehmen musste neue regulatorische Anforderungen für Infrastrukturänderungsauditierung erfüllen. Ihr Legacy-Ansatz — manuelle Änderungen, angewendet von einzelnen Ingenieuren über Cloud-Konsolenzugriff — produzierte keine nutzbare Änderungsspur jenseits roher Cloud-Anbieter-Audit-Logs, die teuer abzufragen und schwer zu interpretieren waren. Die Migration zu IaC mit verpflichtendem Pull-Request-Review adressierte die Audit-Anforderung direkt: Jede Änderung war jetzt ein überprüftes, genehmigtes, versionskontrolliertes Ereignis mit einer Commit-Nachricht, die die geschäftliche Begründung erklärte. Der Audit-Nachweis für das regulatorische Review wurde zu einem einfachen Export des Git-Logs für das Infrastruktur-Repository.
