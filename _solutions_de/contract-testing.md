---
title: Contract Testing
description: Verifikation, dass Service-Schnittstellen vereinbarten Vertragsänderungen
  entsprechen.
category:
- Dependencies
- Testing
quality_tactics_url: https://qualitytactics.de/en/maintainability/contract-testing/
problems:
- poor-contract-design
- rest-api-design-issues
- graphql-complexity-issues
- high-api-latency
- rate-limiting-issues
- legal-disputes
- rapid-system-changes
- maintenance-bottlenecks
- increased-risk-of-bugs
- increased-bug-count
- vendor-relationship-strain
- schema-evolution-paralysis
- testing-complexity
- abi-compatibility-issues
layout: solution
lang: de
en_slug: contract-testing
related_solutions:
- slug: consumer-driven-contracts
  similarity: 0.85
- slug: design-by-contract
  similarity: 0.8
- slug: api-first-design
  similarity: 0.8
- slug: api-documentation
  similarity: 0.75
- slug: api-first-development
  similarity: 0.75
- slug: test-coverage-strategy
  similarity: 0.75
---

## Description

Contract Testing verifiziert, dass ein Service weiterhin die spezifischen Erwartungen erfüllt, auf die sich jeder seiner Konsumenten tatsächlich verlässt, mittels ausführbarer Tests, die aus diesen Erwartungen abgeleitet sind, statt darauf zu vertrauen, dass eine Schnittstellenänderung nichts nachgelagert bricht. Legacy-Systeme häufen genau die Art impliziter, undokumentierter Integrationsverträge an, die diese Praxis explizit macht — Annahmen über Datenformate und Verhalten, die in Konsumenten eingebacken sind, die niemand vollständig katalogisiert hat. Einen konsumentengetriebenen Ansatz zu übernehmen, bei dem jeder Konsument die Teilmenge des Vertrags definiert, auf die er sich verlässt, und der Anbieter verifiziert, dass er alle erfüllt, ist besonders wertvoll, wenn die vollständige Menge der Konsumenten nicht einmal vorab bekannt ist, und erlaubt einem Anbieter, eine unordentliche Schnittstelle weiterzuentwickeln oder zu bereinigen, ohne zu raten, welche Änderung welchen Aufrufer brechen wird.

## How to Apply ◆

> Legacy-Systeme sind durchsetzt mit impliziten Verträgen — undokumentierten Annahmen darüber, wie Komponenten kommunizieren, welche Datenformate sie austauschen und von welchem Verhalten sie abhängen. Contract Testing macht diese Annahmen explizit und verifizierbar und ermöglicht sichere Änderung von Systemen, in denen die Wellenwirkungen von Änderungen sonst unvorhersehbar sind.

- Identifizieren Sie die kritischsten Integrationsgrenzen im Legacy-System: die Schnittstellen zwischen Komponenten, Services oder Systemen, an denen Änderungen am häufigsten Produktionsausfälle verursachen. Diese Hochrisikogrenzen sind dort, wo Contract Testing den unmittelbarsten Wert liefert, und in Legacy-Systemen sind sie oft am wenigsten dokumentiert und am brüchigsten.
- Implementieren Sie konsumentengetriebene Vertragstests mit Frameworks wie Pact, Spring Cloud Contract oder ähnlichen, zum Technologie-Stack passenden Werkzeugen. Beim konsumentengetriebenen Testen definiert jeder Konsument einer API die Teilmenge des Vertrags, von der er abhängt, und der Anbieter verifiziert, dass er alle Konsumentenerwartungen erfüllt. Dieser Ansatz ist besonders wertvoll in Legacy-Systemen, wo die vollständige Menge der Konsumenten möglicherweise nicht unmittelbar bekannt ist.
- Nutzen Sie bei REST-APIs mit Designproblemen Vertragstests, um das tatsächliche aktuelle Verhalten jedes Endpunkts zu kodifizieren — einschließlich seiner Inkonsistenzen —, bevor Sie versuchen, das Design zu verbessern. Dieser „Charakterisierungsvertrag"-Ansatz stellt sicher, dass Standardisierungsbemühungen nicht versehentlich bestehende Konsumenten brechen, die vom aktuellen Verhalten abhängen, selbst wenn dieses Verhalten schlecht gestaltet ist.
- Wenden Sie Schemavalidierung für GraphQL-APIs an, um Abfragekomplexitätslimits, Tiefenbeschränkungen und erforderliche Feldverträge auf Schemaebene durchzusetzen. Vertragstests für GraphQL sollten verifizieren, dass Abfragen, die bestimmte Felder konsumieren, weiterhin erwartete Antwortformen erhalten, und dass Komplexitätslimits konsistent durchgesetzt werden.
- Beziehen Sie Performance-Verträge neben funktionalen Verträgen ein: Spezifizieren Sie erwartete Antwortzeitgrenzen, Rate-Limiting-Verhalten und Payload-Größenlimits als Teil der Vertragsdefinition. Wenn hohe API-Latenz oder Rate-Limiting-Probleme Konsumenten beeinträchtigen, machen Vertragstests mit Performance-Assertions diese Verstöße vor dem Deployment erkennbar statt nach Produktionsvorfällen.
- Nutzen Sie Vertragstests als Grundlage für rechtliche Vereinbarungen, indem Sie technische Vertragsspezifikationen in Sprache übersetzen, auf die nicht-technische Stakeholder und Rechtsteams verweisen können. Wenn Streitigkeiten darüber entstehen, ob ein System seinen Verpflichtungen nachkommt, liefern ausführbare Vertragstests eindeutige Beweise, die die Interpretationskonflikte eliminieren, die Rechtsstreitigkeiten antreiben.
- Integrieren Sie die Ausführung von Vertragstests in CI/CD-Pipelines, sodass jede Änderung an einem Anbieter automatisch gegen alle bekannten Konsumentenverträge verifiziert wird, bevor sie deployt wird. Bei Legacy-Systemen mit langen Release-Zyklen kann dies anfangs bedeuten, Vertragstests nächtlich statt bei jedem Commit auszuführen, mit dem Plan, die Frequenz zu erhöhen, sobald sich die Testsuite stabilisiert.
- Etablieren Sie eine Vertragsversionierungsstrategie, die es Anbietern erlaubt, ihre Schnittstellen weiterzuentwickeln, während Abwärtskompatibilität mit bestehenden Konsumenten erhalten bleibt. Dokumentieren Sie den Versionierungsansatz im Vertrag selbst, einschließlich Deprecation-Zeitplänen und Migrationsleitfäden, sodass Konsumenten ihre Anpassung planen können, statt Breaking Changes in Produktion zu entdecken.

## Tradeoffs ⇄

> Contract Testing verwandelt die impliziten, brüchigen Integrationsannahmen in Legacy-Systemen in explizite, verifizierbare Vereinbarungen, die unabhängige Weiterentwicklung von Komponenten ermöglichen, erfordert aber Koordination zwischen Anbieter- und Konsumententeams, die zuvor möglicherweise nicht über Schnittstellenerwartungen kommuniziert haben.

**Vorteile:**

- Macht REST-API-Designprobleme sichtbar und handhabbar, indem erwartetes Verhalten in ausführbaren Spezifikationen kodifiziert wird, was schrittweise Standardisierung ermöglicht, ohne bestehende Konsumenten zu brechen, die vom aktuellen Verhalten abhängen.
- Reduziert direkt das erhöhte Fehlerrisiko durch Schnittstellenänderungen, indem Vertragsverletzungen vor dem Deployment abgefangen werden, was die Kaskade von Integrationsfehlern verhindert, die Legacy-System-Änderungen häufig auslösen.
- Behebt schlechtes Vertragsdesign, indem eine technische Spezifikation bereitgestellt wird, auf die rechtliche und geschäftliche Vereinbarungen verweisen können, was die Mehrdeutigkeit reduziert, die zu Rechtsstreitigkeiten führt, wenn Parteien uneinig darüber sind, was versprochen wurde.
- Ermöglicht sichere Weiterentwicklung während schneller Systemänderungen, indem verifiziert wird, dass Modifikationen die Verträge bewahren, auf die Konsumenten angewiesen sind, unabhängig davon, wie sich die interne Implementierung ändert.
- Reduziert Wartungsengpässe, indem Entwickler, die nicht die ursprünglichen API-Autoren sind, Anbieterimplementierungen vertrauensvoll ändern können, im Wissen, dass Vertragstests jede unbeabsichtigte Verhaltensänderung abfangen.

**Kosten und Risiken:**

- Konsumentengetriebenes Contract Testing erfordert Zusammenarbeit zwischen Teams, die zuvor möglicherweise nicht koordiniert haben, und in Organisationen mit vielen unabhängigen Konsumenten ist das Sammeln und Pflegen aller Konsumentenverträge ein laufender Aufwand.
- Vertragstests, die zu eng an Implementierungsdetails statt an Verhaltensverträgen gekoppelt sind, werden brüchig und erfordern konstante Pflege, was Overhead ohne proportionale Sicherheit hinzufügt.
- Performance-Verträge sind inhärent umgebungsabhängig und können in CI-Umgebungen, die nicht den Produktions-Performance-Eigenschaften entsprechen, falsch-positive Ergebnisse erzeugen, was sorgfältige Kalibrierung der Performance-Assertions erfordert.
- Legacy-Systeme ohne bestehende API-Dokumentation erfordern erheblichen Aufwand, um aktuelles Verhalten zu entdecken und zu kodifizieren, bevor Vertragstests geschrieben werden können, und der Entdeckungsprozess selbst kann Inkonsistenzen offenbaren, die schwer aufzulösen sind.
- Übermäßiges Vertrauen auf Contract Testing kann ein falsches Sicherheitsgefühl erzeugen, wenn die Verträge nicht die volle Bandbreite realer Nutzungsmuster abdecken, einschließlich Randfällen und Fehlerszenarien, die in Legacy-Integrationen häufig sind.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Contract Testing die spezifischen Integrationsherausforderungen in Legacy-Systemen mit schlecht dokumentierten und inkonsistent gestalteten Schnittstellen adressiert.

Ein Zahlungsabwicklungsunternehmen pflegt eine Legacy-API, die von 40 externen Händlerintegrationen konsumiert wird, jede über das letzte Jahrzehnt gegen eine undokumentierte Schnittstelle gebaut, die sich durch Ad-hoc-Änderungen weiterentwickelt hat. Als das Team versucht, das inkonsistente Fehlerantwortformat der API zu standardisieren — manche Endpunkte geben Fehler als JSON-Objekte zurück, andere als reine Textstrings, und einer gibt XML zurück —, entdecken sie, dass Händlerintegrationen Fehler auf formatspezifische Weise parsen. Mit der Implementierung konsumentengetriebener Vertragstests mit Pact bittet das Team jedes Händler-Integrationsteam, einen Vertrag einzureichen, der die erwarteten Fehlerformate definiert. Die resultierenden Verträge zeigen, dass 12 Händler explizit vom Klartext-Fehlerformat für einen bestimmten Endpunkt abhängen. Das Team implementiert einen Migrationsplan, der das standardisierte JSON-Format als Standard einführt, während Abwärtskompatibilität für diese 12 Konsumenten erhalten bleibt, mit einem im Vertrag dokumentierten sechsmonatigen Deprecation-Zeitplan. Ohne Contract Testing hätte die Standardisierung ein Drittel der Händlerintegrationen in Produktion gebrochen.

Ein Gesundheits-Softwareanbieter steht vor einem Rechtsstreit mit einem Krankenhauskunden darüber, ob die gelieferte API „branchenübliche Antwortzeiten" erfüllt, wie im Vertrag spezifiziert. Der Anbieter behauptet, die API antworte im Durchschnitt innerhalb von 200ms, während das Krankenhaus in seiner Umgebung 3-Sekunden-Antwortzeiten meldet. Der Streit hat vier Monate rechtlicher Aufmerksamkeit verbraucht und bedroht die Geschäftsbeziehung. Nach der Implementierung von Vertragstests mit Performance-Assertions mit spezifischen Perzentil-Zielen (p50 unter 200ms, p95 unter 500ms, p99 unter 2 Sekunden), die gegen eine Referenzumgebung laufen, einigen sich beide Parteien darauf, die Vertragstestergebnisse als objektives Maß der Konformität zu nutzen. Die Tests zeigen, dass die API Performance-Ziele für einfache Abfragen erfüllt, sie aber für komplexe Patientenaktenabrufe wegen N+1-Abfragemustern überschreitet. Die konkreten Testergebnisse verwandeln den Rechtsstreit in einen technischen Abhilfeplan, und zukünftige Verträge verweisen auf ausführbare Vertragstests statt auf vage Performance-Sprache.
