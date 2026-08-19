---
title: Anwendungsportfolio-Inventar
description: Pflege eines einzigen Verzeichnisses darüber, welche Systeme existieren,
  was sie tun, wer sie besitzt, wovon sie abhängen und in welchem Zustand sie sind.
category:
- Architecture
- Management
- Dependencies
problems:
- system-integration-blindness
- hidden-dependencies
- technology-stack-fragmentation
- obsolete-technologies
- unclear-documentation-ownership
- lack-of-ownership-and-accountability
- vendor-dependency
- knowledge-gaps
- monitoring-gaps
- legacy-system-documentation-archaeology
- shared-dependencies
- high-maintenance-costs
- accumulated-decision-debt
- communication-risk-outside-project
- dependency-on-supplier
- information-decay
- legacy-configuration-management-chaos
- legal-disputes
- modernization-roi-justification-failure
- technical-architecture-limitations
- vendor-relationship-strain
- retention-obligations-block-change
- voided-vendor-support
layout: solution
lang: de
en_slug: application-portfolio-inventory
related_solutions:
- slug: system-decommissioning
  similarity: 0.7
- slug: dependency-management-strategy
  similarity: 0.7
- slug: clear-roles-and-ownership
  similarity: 0.7
- slug: technology-radar
  similarity: 0.7
- slug: clear-ownership-model
  similarity: 0.7
- slug: change-impact-analysis
  similarity: 0.7
---

## Description

Ein Anwendungsportfolio-Inventar ist ein einziges gepflegtes Verzeichnis der Systeme, die eine Organisation betreibt: was jedes tut, wer es besitzt, welche Technologie es nutzt, wovon es abhängt und was von ihm abhängt, seine geschäftliche Kritikalität und seinen Zustand. Es klingt nach Verwaltung, und es ist die Voraussetzung für nahezu jede strategische Entscheidung über eine Legacy-Landschaft. Organisationen jeden Alters können routinemäßig grundlegende Fragen nicht beantworten — wie viele Systeme haben wir, welche handhaben personenbezogene Daten, was würde brechen, wenn diese Datenbank nicht verfügbar wäre — und jeder Modernisierungsplan, jede Risikobewertung und Auswirkungsanalyse beginnt damit, die Antwort für diesen Anlass teilweise zu rekonstruieren. Das Inventar ersetzt ein Dutzend teilweiser Rekonstruktionen durch ein einziges gepflegtes Verzeichnis. Sein Wert ist proportional dazu, wie unorganisiert die Landschaft ist, was bedeutet, dass es genau dort am wertvollsten ist, wo es am schwierigsten aufzubauen ist.

## How to Apply ◆

> Die Systeme, die in der mentalen Landkarte jeder Organisation fehlen, sind die, an die seit Jahren niemand mehr gedacht hat, was ein starker Prädiktor dafür ist, welche gleich ein Problem verursachen werden.

- **Entdecken statt befragen.** Beginnen Sie mit Infrastruktur-Inventaren, Netzwerk-Traffic, DNS-Einträgen, Zertifikatsregistern, Authentifizierungsprotokollen, Lizenzunterlagen und Firewall-Regeln. Teams zu fragen, was sie betreiben, findet, woran sie sich erinnern; automatisierte Entdeckung findet, was sie vergessen haben.
- **Halten Sie das Verzeichnis bewusst klein.** Zehn bis fünfzehn Felder, nicht mehr: Name, Zweck in einem Satz, besitzendes Team, Technologie, Kritikalität, vor- und nachgelagerte Abhängigkeiten, Datenklassifizierung, Support-Status und Datum der letzten Überprüfung. Ambitionierte Schemata produzieren Inventare, die zu neunzig Prozent leer sind.
- **Erfassen Sie Abhängigkeiten in beide Richtungen.** Was dieses System aufruft und was es aufruft. Die eingehende Richtung ist schwieriger festzustellen und diejenige, die für Auswirkungsanalyse und Außerbetriebnahme benötigt wird, weshalb sie üblicherweise die fehlende ist.
- **Verlangen Sie ein benanntes besitzendes Team pro System**, und behandeln Sie jedes System ohne eines als Befund statt als Lücke im Verzeichnis. Nicht besessene Systeme sind dort, wo sich Vorfälle und ungepatchte Schwachstellen konzentrieren.
- **Fügen Sie die operative Realität hinzu** — Support-Status, End-of-Support-Daten, letztes Patch, ob Monitoring existiert, ob ein Recovery-Verfahren getestet wurde. Das ist, was ein Inventar von einem Katalog in ein Risikoregister verwandelt.
- **Binden Sie es an einen Prozess, der es aktuell hält**, oder akzeptieren Sie, dass es innerhalb eines Jahres veraltet sein wird. Das Onboarding eines neuen Systems, die Außerbetriebnahme eines Systems und jeder Eigentümerwechsel müssen das Verzeichnis aktualisieren, und die Aktualisierung muss Teil des Prozesses sein statt eines Akts der Tugend.
- **Überprüfen Sie periodisch eine rotierende Teilmenge**, statt einen vollständigen Refresh zu versuchen. Zwanzig Einträge pro Quartal halten ein 200-System-Inventar mit bescheidenem Aufwand ungefähr aktuell; eine jährliche Vollüberprüfung wird geplant, verschoben und nie durchgeführt.
- **Machen Sie es zur akzeptierten einzigen Quelle.** Ein Inventar, das mit drei Tabellen in verschiedenen Abteilungen konkurriert, ist eine vierte Tabelle. Die Konsolidierung der bestehenden Teilverzeichnisse ist üblicherweise die erste und politisch schwierigste Arbeit.
- **Veröffentlichen Sie es breit.** Sein Wert kommt daraus, konsultiert zu werden, und es wird nur konsultiert, wenn Menschen wissen, dass es existiert und es durchsuchen können, ohne um Erlaubnis zu fragen.
- **Nutzen Sie es, um Entscheidungen zu treiben**, nicht nur um zu beschreiben: Kandidaten für Außerbetriebnahme, nicht besessene Systeme, nicht unterstützte Technologie und Single Points of Failure fallen alle direkt aus den obigen Feldern heraus.

## Tradeoffs ⇄

> Ein Inventar ist die Grundlage für jede strategische Entscheidung über eine Legacy-Landschaft, aber es aufzubauen ist unglamourös, und es aktuell zu halten erfordert dauerhafte Disziplin.

**Vorteile:**

- Auswirkungsanalyse, Außerbetriebnahme und Modernisierungsplanung starten alle von einer bekannten Grundlage statt einer jedes Mal wiederholten Teilrekonstruktion.
- Nicht besessene Systeme und nicht unterstützte Technologien werden sichtbar, und dies sind konsistent die Stellen, an denen sich Vorfälle und Sicherheitsrisiken konzentrieren.
- Bidirektionale Abhängigkeitsverzeichnisse machen den Explosionsradius einer Änderung oder eines Ausfalls im Voraus bewertbar.
- Compliance- und Audit-Fragen — wo personenbezogene Daten leben, was unter eine gegebene Regulierung fällt — werden in Stunden statt Wochen beantwortbar.
- Kandidaten für die Stilllegung tauchen automatisch aus der Kombination von geringer Kritikalität, nicht unterstützter Technologie und fehlendem Eigentümer auf.

**Kosten und Risiken:**

- Die anfängliche Entdeckung ist erhebliche Arbeit ohne unmittelbares Ergebnis, und es ist schwer, sie gegen alles zu finanzieren, das ein Feature liefert.
- Inventare veralten schnell, und ein veraltetes ist gefährlich, weil auf seiner Basis Entscheidungen getroffen werden — ein Eintrag, der einen Eigentümer nennt, der vor zwei Jahren gegangen ist, ist schlimmer als ein leeres Feld.
- Übermäßig ambitionierte Schemata brechen unter ihrem eigenen Gewicht zusammen und produzieren ein größtenteils leeres Verzeichnis, dem niemand vertraut oder das niemand pflegt.
- Eigentümerfelder machen Verantwortlichkeit explizit, was in Organisationen auf Widerstand stößt, in denen als Eigentümer benannt zu werden bedeutet, ein Problem ohne Ressourcen zu erben.
- Das Inventar kann zum Selbstzweck werden und Aufwand in Pflege und Berichterstattung verbrauchen, der die Entscheidungen übersteigt, die es informiert.

## How It Could Be

Eine Organisation glaubte, sie betreibe ungefähr 60 Anwendungen. Entdeckung aus Zertifikatsverzeichnissen, Netzwerk-Traffic-Protokollen und Lizenzdaten fand 143 verschiedene laufende Systeme, einschließlich einer kundenzugewandten Webanwendung, von der niemand in der aktuellen Organisation wusste — ein Legacy-Selbstbedienungsportal aus einer sechs Jahre zurückliegenden Akquisition, immer noch vom Internet aus erreichbar, immer noch gegen ein Verzeichnis authentifizierend und zuletzt 2019 gepatcht. Es hatte keinen Eigentümer, kein Monitoring und keinen Eintrag in irgendeiner Liste. Seine Entdeckung war das stärkste Argument für die Finanzierung der Inventararbeit, die für die verbleibenden 142 Systeme über zwei Quartale abgeschlossen wurde. Neunzehn Systeme stellten sich als ohne identifizierbaren Eigentümer heraus, und elf liefen mit Technologie jenseits des Anbieter-Supports.

Die bidirektionalen Abhängigkeitsverzeichnisse veränderten die Vorfallreaktion der Organisation. Ein Datenbankcluster benötigte Notfallwartung, und der bisherige Ansatz wäre gewesen, alle zu benachrichtigen und zu hoffen. Das Inventar listete neun Systeme mit einer verzeichneten Abhängigkeit von diesem Cluster auf, von denen drei als geschäftskritisch klassifiziert waren. Die Wartung wurde speziell um diese drei herum geplant, wobei ihre besitzenden Teams im Voraus einbezogen wurden. Während des Zeitfensters fiel ein zehntes System aus — eines, dessen Abhängigkeit undokumentiert war, hinzugefügt acht Monate zuvor von einem Team, das das Verzeichnis nicht aktualisiert hatte. Dieser Ausfall wurde zum Grund, warum Abhängigkeitsaktualisierungen zu einem verpflichtenden Schritt im Änderungsprozess gemacht wurden statt zu einer Höflichkeit — die Art von Durchsetzung, die ein Inventar braucht und selten bekommt, bevor etwas zeigt, warum.
