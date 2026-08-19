---
title: Anti-Corruption Layer
description: Schutz bestehender Systeme vor negativen Einflüssen externer Systeme.
category:
- Architecture
- Dependencies
problems:
- architectural-mismatch
- poor-interfaces-between-applications
- integration-difficulties
- vendor-dependency
- vendor-dependency-entrapment
- vendor-lock-in
- technology-lock-in
- legacy-api-versioning-nightmare
- api-versioning-conflicts
- shared-dependencies
- cross-system-data-synchronization-problems
- breaking-changes
- dependency-on-supplier
- strangler-fig-pattern-failures
- shared-database
layout: solution
lang: de
en_slug: anti-corruption-layer
related_solutions:
- slug: strangler-fig-pattern
  similarity: 0.7
- slug: transactions
  similarity: 0.7
- slug: event-driven-architecture
  similarity: 0.7
- slug: modularization-and-bounded-contexts
  similarity: 0.7
- slug: architecture-decision-records
  similarity: 0.7
- slug: ubiquitous-language
  similarity: 0.7
---

## Description

Ein Anti-Corruption Layer ist ein dediziertes Grenzmodul, das zwischen einem Legacy- oder externen System und dem Rest der Codebasis sitzt und das Datenmodell, das Vokabular und die Fehlersemantik des externen Systems in saubere Domänentypen übersetzt, bevor irgendetwas anderes in der Anwendung sie sehen darf. Anders als ein einfacher Adapter ist sein Zweck explizit defensiv: Er existiert, um zu verhindern, dass die Eigenheiten eines Legacy-Systems — obskure Statuscodes, Datensatzformate fester Breite, inkonsistente Identifikatoren, fehlangepasste Terminologie — in das Design des neuen oder umgebenden Systems einsickern und es korrumpieren. Diese Unterscheidung zählt am meisten in der Legacy-Modernisierung, wo neue Services häufig weiterhin mit einem alten Mainframe, einem Anbietersystem oder mehreren redundanten Legacy-Quellen gleichzeitig sprechen müssen, jede mit eigenen Konventionen, die keinen natürlichen Platz in einem sauberen Domänenmodell haben. Durch die Konzentration aller legacy-spezifischen Übersetzungen und Fehlerbehandlung innerhalb des ACL bleibt der Rest des Systems intern konsistent, selbst wenn er von mehreren Legacy-Quellen abhängt, die sich untereinander widersprechen, und der ACL wird zum einzigen, gut verstandenen Ort, an dem Legacy-Verhaltensänderungen absorbiert werden müssen. Er bietet außerdem eine natürliche Naht für eine Strangler-Fig-Migration, da neue Funktionalität gegen das saubere Domänenmodell gebaut werden kann, während der ACL weiterhin zum Legacy-System überbrückt, bis dieses schließlich abgeschaltet wird. Das Muster erfordert anhaltende Investition zum Bau und zur Wartung pro Integration, und wenn Entwickler ihn aus Bequemlichkeit umgehen oder ihn zu einem Ad-hoc-Datenspeicher eigener Art wachsen lassen, kann er die genau von ihm eingedämmte Komplexität neu erschaffen.

## How to Apply ◆

> Bei Legacy-Integrationsarbeit ist der Anti-Corruption Layer die primäre Verteidigung dagegen, dass das Datenmodell, die Namenskonventionen und Fehlercodes eines alten Systems in neuen Code einsickern.

- Identifizieren Sie alle Punkte, an denen Ihr neuer oder modernisierter Code mit einem Legacy-System sprechen muss — Mainframe-Schnittstellen, COBOL-Copybook-Formate, alternde SOAP-Endpunkte, proprietäre Flatfile-Feeds — und machen Sie jeden zu einer expliziten ACL-Grenze statt eines direkten Aufrufs.
- Bauen Sie ein dediziertes Modul oder Paket für jeden ACL; lassen Sie niemals Domänencode direkt externe API-Clients importieren, weil selbst ein direkter Import die Korruption startet.
- Implementieren Sie einen Übersetzer für jedes externe Konzept: Bilden Sie die undurchsichtigen Statuscodes, abgekürzten Feldnamen und Datensätze fester Breite des Legacy-Systems auf bedeutungsvolle Domänentypen ab, bevor irgendetwas anderes in der Codebasis sie sieht.
- Schreiben Sie Integrationstests mit aufgezeichneten echten Antworten des Legacy-Systems; wenn sich die Legacy-Schnittstelle ändert (und das wird sie), werden diese Tests fehlschlagen, bevor Produktions-Traffic betroffen ist.
- Nutzen Sie den ACL zur Zentralisierung aller legacy-spezifischen Fehlerbehandlung — Retries, Timeouts, CICS-ABEND-Codes, DB2-SQL-Codes —, sodass neue Services nie wissen müssen, was ein `SQLCODE -911` bedeutet.
- Wenn das Legacy-System eine neue Schnittstellenversion neben der alten veröffentlicht, fügen Sie einen zweiten Adapter und Übersetzer innerhalb des ACL hinzu und nutzen Sie einen Konfigurationsschalter zur Kontrolle, welche Version aktiv ist; dies erlaubt parallelen Betrieb, ohne eine einzige Zeile Domänencode zu ändern.
- Fügen Sie einen Circuit Breaker innerhalb des ACL für Legacy-Backends hinzu, die langsam oder unzuverlässig sind, wobei zwischengespeicherte oder degradierte Antworten zurückgegeben werden statt Latenz an moderne Services weiterzugeben.
- Überwachen Sie Übersetzungsfehler als separate operative Metrik; eine steigende Fehlerrate signalisiert fast immer eine undokumentierte Änderung in der Ausgabe des Legacy-Systems.

## Tradeoffs ⇄

> Der ACL fügt eine Codeschicht zum Schreiben und Warten hinzu, aber in Legacy-Kontexten wird dieser Aufwand fast immer durch den Schutz aufgewogen, den er gegen die hartnäckigste Verfallsform des Systems bietet — angehäufte Modellkorruption.

**Vorteile:**

- Hält die neue Codebasis sauber und intern konsistent, selbst wenn sie mit mehreren Legacy-Systemen kommunizieren muss, die jeweils unterschiedliche Feldnamen, Datenformate und Statuscodes nutzen.
- Konzentriert alles legacy-spezifische Wissen an einem Ort, was es weit einfacher macht, es zu verstehen, zu testen und schließlich zu entfernen, wenn das Legacy-System abgeschaltet wird.
- Ermöglicht den Ersatz oder das Upgrade von Legacy-Backends, ohne Domänencode zu berühren — nur der Adapter und Übersetzer im ACL müssen sich ändern.
- Bietet eine natürliche Naht für Strangler-Fig-Migration: Neue Funktionalität kann zum modernen System hinzugefügt werden, während der ACL weiterhin die Lücke zu Legacy-Komponenten überbrückt.
- Schützt das Team vor undokumentierten Legacy-Verhaltensänderungen, indem sie als ACL-Validierungs- oder Übersetzungsfehler statt stiller Datenkorruption nachgelagert auftauchen.

**Kosten und Risiken:**

- Jede Legacy-Integration erfordert das Design, den Bau und die Wartung ihres eigenen ACL, was Entwicklungsaufwand hinzufügt, den Teams unter Modernisierungsdruck möglicherweise unterschätzen.
- Wenn sich das Modell des Legacy-Systems häufig ändert (häufig bei Systemen, die noch aktiv gewartet werden), erfordern die Übersetzer innerhalb des ACL konstante Aktualisierungen und können zu einem Engpass werden.
- Entwickler, die mit dem Muster nicht vertraut sind — besonders solche, die mit dem Legacy-System aufgewachsen sind — könnten den ACL aus Bequemlichkeit umgehen und die Modellkorruption neu erschaffen, die die Schicht verhindern sollte.
- Ein schlecht designter ACL, der Zustand ansammelt oder zu groß wird, kann selbst zu einem Legacy-Problem werden und die Komplexität erben, gegen die er schützen sollte.
- Latenz wird für jeden grenzüberschreitenden Aufruf hinzugefügt; in Szenarien mit hohem Integrationsdurchsatz muss dieser Overhead gemessen und verwaltet werden.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie das ACL-Muster reale Integrationszwänge in Legacy-Modernisierungsprogrammen löst.

Eine Einzelhandelsbank ersetzte ihr kundenseitiges Kreditantrags-Portal, während sie ihr Mainframe-basiertes Kernbanksystem unverändert beließ. Der Mainframe sprach in EBCDIC-kodierten Datensätzen fester Breite mit Feldnamen wie `CUST-NO`, `LN-AMT-APPRVD` und `DT-ORIG` und nutzte zweistellige Begründungscodes zur Signalisierung von Genehmigungsergebnissen. Statt diese Strukturen direkt in das Domänenmodell des neuen Portals abzubilden, baute das Team einen Kredit-Gateway-ACL, der Mainframe-Datensätze in ordentliche `LoanApplication`- und `CreditDecision`-Domänenobjekte übersetzte. Als das Mainframe-Team später das Datensatzlayout reorganisierte, um einen neuen Produkttyp zu unterstützen, musste nur der Übersetzer des ACL geändert werden — keine der Geschäftslogik des Portals war betroffen.

Ein Versicherungsunternehmen, das ein neues Schadensmanagementsystem mit drei Legacy-Policenverwaltungssystemen integrierte, entdeckte, dass jedes System einen anderen Identifikator für denselben Versicherungsnehmer nutzte. Eines nutzte eine neunstellige Kontonummer, ein anderes eine Sozialversicherungsnummer, und das dritte einen internen sequenziellen Schlüssel. Das Team erstellte einen separaten ACL für jedes System, wobei jeder seinen lokalen Identifikator auf die kanonische `PolicyholderId` auflöste, die im gesamten neuen System genutzt wurde. Die ACLs normalisierten außerdem die wild unterschiedlichen Schadensstatus-Vokabulare — „PEND", „OPEN", „AO", „CLD" aus drei verschiedenen Systemen — in eine einzige `ClaimStatus`-Enumeration. Als Ermittler später einen Schaden über alle drei Systeme hinweg nachverfolgen mussten, boten die ACLs einen kontrollierten, dokumentierten Übersetzungspfad statt eines Labyrinths aus Ad-hoc-String-Vergleichen.

Ein Logistikunternehmen, das seine Paketverfolgungsplattform modernisierte, konsumierte Ereignis-Feeds von vier verschiedenen Frachtführer-APIs. Jeder Frachtführer repräsentierte Sendungsereignisse unterschiedlich: Einer gab ISO-Zeitstempel zurück, ein anderer nutzte Unix-Epoch-Millisekunden, ein dritter sendete Daten in `MM/DD/YYYY`-Ortszeit. Ein Frachtführer beschrieb denselben physischen Zustand — Paket beim Zoll zurückgehalten — mit drei unterschiedlichen Ereigniscodes je nach Land, in dem sich das Paket befand. Statt frachtführerspezifisches Parsing über den gesamten Tracking-Service zu verstreuen, baute das Team einen Frachtführer-ACL pro Integration, der alle Ereignisse in ein kanonisches `TrackingEvent`-Domänenobjekt normalisierte, bevor sie das System betraten. Die Einbindung eines fünften Frachtführers später erforderte nur das Hinzufügen eines neuen ACL-Adapters; der Rest der Plattform blieb unberührt.
